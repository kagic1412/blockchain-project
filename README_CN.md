# Elliptic++ AML 检测项目

图神经网络（GNN）与树模型在区块链交易反洗钱（AML）检测中的对比研究。

## 项目概述

本项目对比了 GraphSAGE 图神经网络与 LightGBM 树模型在检测比特币非法交易上的效果，分析了 GNN 在该任务上表现不佳的原因，并使用 Optuna 进行超参数优化。

## 数据集

**注意**：由于文件大小限制，数据集文件未包含在仓库中，请从官方渠道下载。

### 下载方式

1. 从 [Elliptic++ 数据集仓库](https://github.com/git-disl/EllipticPlusPlus/tree/main/Transactions%20Dataset) 下载数据集
2. 下载数据集文件
3. 将文件放入 `data/raw/` 目录，命名为：
   - `txs_classes.csv`
   - `txs_edgelist.csv`
   - `txs_features.csv`

### 数据集概况

[Elliptic++](https://www.elliptic.co/) 数据集包含 203,769 笔比特币交易：
- 每笔交易 **166 个特征**（局部特征和聚合特征）
- **图结构**：234,355 条边，展示交易资金流向
- **时序划分**：训练集（时间步 ≤ 34），测试集（时间步 > 34）
- **类别分布**：1=非法（4,545），2=合法（42,019），3=未知（157,205）

## 项目目录结构

```
blockchain/
├── data/
│   ├── raw/                      # 原始数据文件
│   │   ├── txs_classes.csv       # 交易类别标签
│   │   ├── txs_edgelist.csv     # 交易图边列表
│   │   └── txs_features.csv      # 交易特征
│   ├── processed/                # 处理后的数据
│   │   └── txs_merge_with_network.csv  # 最终数据集
│   └── output/                   # 模型输出
│
├── scripts/
│   ├── data_processing/          # 数据处理脚本
│   │   ├── extract_network_features.py
│   │   ├── merge_data.py
│   │   └── feature_analysis.py
│   └── models/                   # 模型训练脚本
│       ├── baseline_model.py     # LightGBM 基线模型
│       ├── graphsage_aml.py      # GraphSAGE GNN
│       ├── graphsage_optuna.py    # 超参数优化
│       └── summary.py             # 模型对比
│
├── analysis/                      # 分析结果输出
├── results/                       # 实验结果
├── README.md                     # 英文文档
├── README_CN.md                  # 本文件（中文文档）
└── .gitignore
```

## 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/elliptic-aml.git
cd elliptic-aml

# 创建 conda 环境
conda create -n pytorch_env python=3.9
conda activate pytorch_env

# 安装 PyTorch
pip install torch torchvision torchaudio

# 安装 PyTorch Geometric（需要先安装 PyTorch）
pip install pyg-nightly torch-geometric

# 安装其他依赖
pip install -r requirements.txt
```

## 快速开始

### 1. 数据准备

```bash
# 合并特征与标签
python scripts/data_processing/merge_data.py

# 提取网络特征
python scripts/data_processing/extract_network_features.py

# 合并网络特征
python scripts/data_processing/merge_network_features.py
```

### 2. 运行模型

```bash
# LightGBM 基线模型
python scripts/models/baseline_model.py

# GraphSAGE GNN（F1 优化参数）
python scripts/models/graphsage_aml.py

# GNN + Optuna 超参数搜索
python scripts/models/graphsage_optuna.py
```

### 3. 模型对比

```bash
python scripts/summary.py
```

## 性能结果

| 模型 | Recall | Precision | F1-Score |
|------|--------|-----------|----------|
| LightGBM (189 特征) | **0.73** | **0.90** | **0.81** |
| GraphSAGE (50 特征, F1优化) | 0.62 | 0.65 | 0.63 |
| GraphSAGE (50 特征, Recall优化) | 0.87 | 0.16 | 0.27 |

### 主要发现

树模型在所有指标上均显著优于 GNN，主要问题包括：

1. **未知节点干扰**：77% 的节点标签未知，污染了消息传递过程
2. **时序分布偏移**：图结构随时间演变
3. **训练策略简陋**：无验证集、无早停机制

## 模型配置

### LightGBM 基线
```python
LGBMClassifier(
    is_unbalance=True,
    n_estimators=200,
    learning_rate=0.1,
    max_depth=10,
    num_leaves=31
)
```

### GraphSAGE（F1 优化）
```python
GraphSAGE(
    in_channels=50,      # Top 50 特征
    hidden_channels=64,
    out_channels=2,
    dropout=0.476,
    aggr='max'
)
# weight_1 = 4.56
# lr = 0.039
# epochs = 250
```

## 文档

- English: See [`README.md`](README.md)
- 中文文档：本文件

## 许可证

本项目仅供研究使用。Elliptic++ 数据集由 Elliptic 提供，遵循其自有许可证。

## 致谢

- [Elliptic++](https://www.elliptic.co/) 数据集
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [LightGBM](https://lightgbm.readthedocs.io/)
