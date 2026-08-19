# IMDb 情感分类 - Transformer 实现

## 项目简介
使用纯 PyTorch 手写 Transformer Encoder，在 IMDb 影评数据集上进行情感二分类（正面/负面）。

## 主要结果
- 最终测试准确率：**83.15%**
- 使用 GloVe 100d 预训练词向量
- 早停策略 + Dropout 正则化

## 技术栈
- PyTorch
- HuggingFace Datasets
- torchtext（词向量加载）
- GloVe 预训练词向量

## 项目结构
- PositionalEncoding
- MultiHeadAttention
- FeedForward
- EncoderBlock
- Encoder
- TransformerClassifier

## 实验记录
