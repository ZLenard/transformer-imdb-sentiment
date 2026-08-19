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
| 配置 | 测试准确率 | 备注 |
| :--- | :--- | :--- |
| 随机 Embedding + 大模型 (dropout=0.1) + 5 epochs | 80.93% | 过拟合明显 |
| 随机 Embedding + 小模型 (dropout=0.5) + 5 epochs | 80.18% | 欠拟合，容量不足 |
| 随机 Embedding + 中等模型 (dropout=0.35) + 5 epochs | 78.54% | 学不进去，收敛慢 |
| 随机 Embedding + 中等模型 (dropout=0.35) + 12 epochs | 79.34% | 训练更久，但提升有限 |
| **GloVe 100d + 中等模型 (dropout=0.35) + 12 epochs** | **83.15%** | ✅ 有效突破 |

## 备注
本项目是 Transformer 学习实践的产物。核心模块（MultiHeadAttention、PositionalEncoding、EncoderBlock 等）在理解原理后参考 PyTorch 官方文档和社区示例，独立复现并调试通过。
