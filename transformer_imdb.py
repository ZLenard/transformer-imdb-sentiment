import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import GloVe
from datasets import load_dataset
from collections import Counter
import math
import os

# ---------- 1. Transformer 组件（保持不变） ----------
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

    def forward(self, q, k=None, v=None, mask=None):
        batch_size = q.size(0)
        seq_len_q = q.size(1)

        if k is None:
            k = q
        if v is None:
            v = q

        seq_len_k = k.size(1)
        seq_len_v = v.size(1)

        Q = self.W_q(q).view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(k).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(v).view(batch_size, seq_len_v, self.num_heads, self.head_dim).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 1, -1e9)

        attn = F.softmax(scores, dim=-1)
        out = attn @ V

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.embed_dim)
        out = self.W_o(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, max_len, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.positional_encoding = PositionalEncoding(embed_dim, max_len)
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.feed_forward = FeedForward(embed_dim, ff_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.positional_encoding(x)
        attn_out = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class Encoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers, max_len, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, ff_dim, max_len, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_len, num_classes, dropout=0.1, pretrained_weights=None):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if pretrained_weights is not None:
            self.embedding.weight.data.copy_(pretrained_weights)
        self.encoder = Encoder(embed_dim, num_heads, ff_dim, num_layers, max_len, dropout)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# ---------- 2. 数据准备 ----------
def build_vocab():
    tokenizer = get_tokenizer('basic_english')
    train_ds = load_dataset('imdb', split='train')
    test_ds = load_dataset('imdb', split='test')

    counter = Counter()
    for sample in train_ds:
        tokens = tokenizer(sample['text'])
        counter.update(tokens)

    vocab = {'<pad>': 0, '<unk>': 1}
    for word, _ in counter.most_common(20000 - len(vocab)):
        if word not in vocab:
            vocab[word] = len(vocab)

    return vocab, tokenizer, train_ds, test_ds

class IMDbDataset(Dataset):
    def __init__(self, hf_dataset, vocab, tokenizer, max_len=128):
        self.data = []
        pad_id = vocab['<pad>']
        unk_id = vocab['<unk>']
        for sample in hf_dataset:
            tokens = tokenizer(sample['text'])[:max_len]
            ids = [vocab.get(token, unk_id) for token in tokens]
            padded = ids + [pad_id] * (max_len - len(ids))
            self.data.append((torch.tensor(padded, dtype=torch.long), sample['label']))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ---------- 3. 加载 GloVe 词向量 ----------
def load_glove_embeddings(vocab, embed_dim):
    # 加载 GloVe 向量（首次会下载约 800MB，请保持网络畅通）
    glove = GloVe(name='6B', dim=embed_dim)  # 支持 50, 100, 200, 300 维
    pretrained_weights = torch.zeros(len(vocab), embed_dim)
    for word, idx in vocab.items():
        if word in glove.stoi:
            pretrained_weights[idx] = glove[word]
        else:
            # 不在词表中的词，用均匀分布初始化
            pretrained_weights[idx] = torch.randn(embed_dim) * 0.1
    return pretrained_weights

# ---------- 4. 训练与评估 ----------
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return total_loss / len(dataloader), correct / total

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return total_loss / len(dataloader), correct / total

# ---------- 5. 主程序 ----------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 超参数（使用 100 维 GloVe）
    embed_dim = 100
    num_heads = 5
    ff_dim = 200
    num_layers = 2
    max_len = 128
    num_classes = 2
    batch_size = 64
    epochs = 12
    lr = 0.0003
    weight_decay = 1e-4
    dropout = 0.35

    # 加载数据
    print("加载数据...")
    vocab, tokenizer, train_ds, test_ds = build_vocab()
    print(f"词表大小: {len(vocab)}")

    # 加载 GloVe 词向量
    print("加载 GloVe 预训练词向量（首次需下载约 800MB）...")
    pretrained_weights = load_glove_embeddings(vocab, embed_dim)
    print("词向量加载完成。")

    train_dataset = IMDbDataset(train_ds, vocab, tokenizer, max_len)
    test_dataset = IMDbDataset(test_ds, vocab, tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 创建模型
    model = TransformerClassifier(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        max_len=max_len,
        num_classes=num_classes,
        dropout=dropout,
        pretrained_weights=pretrained_weights
    ).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_test_acc = 0
    patience = 2
    trigger_count = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            trigger_count = 0
            torch.save(model.state_dict(), 'best_model_glove.pt')
        else:
            trigger_count += 1
            if trigger_count >= patience:
                print(f"早停在第 {epoch+1} 个 Epoch，测试准确率不再提升")
                break
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    print(f"\n最佳测试准确率: {best_test_acc:.4f}")
