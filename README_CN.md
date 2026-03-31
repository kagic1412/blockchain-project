# Elliptic++ AML 检测项目

图神经网络（GNN）与树模型在区块链交易反洗钱（AML）检测中的对比研究。

## 项目概述

本项目对比了 GraphSAGE 图神经网络与 LightGBM 树模型在检测比特币非法交易上的效果，分析了 GNN 在该任务上表现不佳的原因，并使用 Optuna 进行超参数优化。

## 数据集

**注意**：由于文件大小限制，数据集文件未包含在仓库中，请从官方渠道下载。

### 下载方式

1. 从 [Elliptic++ 数据集仓库](https://github.com/git-disl/EllipticPlusPlus/tree/main/Transactions%20Dataset) 下载数据集
2. 在项目根目录下创建以下目录结构：
   ```
   data/
   ├── raw/
   ├── processed/
   └── output/
   ```
3. 将下载的文件放入 `data/raw/` 目录，命名为：
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
│   └── visualization_analysis.md  # 网络可视化分析报告
├── results/                       # 实验结果
├── notebook/                      # Jupyter notebooks
│   └── Elliptic_AML_Detection_Project_EN.ipynb  # 完整分析 Notebook（可直接查看运行结果）
├── visualization_analysis.md      # 网络可视化分析报告（英文版）
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

## 快速查看结果（推荐）

无需安装环境、无需下载数据，直接在 GitHub 或本地打开 Jupyter Notebook 即可查看完整运行结果。

### notebook/Elliptic_AML_Detection_Project_EN.ipynb

本项目包含一个完整的可交互分析 notebook，涵盖：

- **完整数据分析流程**：数据加载 → 特征提取 → 模型训练 → 结果对比
- **所有运行结果**：包括分类报告、特征重要性、模型对比图表
- **英文说明文档**：便于国际交流

**使用方法**：

1. **GitHub 在线查看**：进入 `notebook/` 文件夹，点击 `.ipynb` 文件，GitHub 会自动渲染所有代码和输出
2. **本地运行**：
   ```bash
   jupyter notebook notebook/Elliptic_AML_Detection_Project_EN.ipynb
   ```
   如果数据文件不存在，notebook 会提示下载链接

> **提示**：上传 GitHub 后，他人可以直接看到所有运行结果，无需自己运行代码。

---

## 快速开始（本地运行）

如果需要在自己的数据集上实验，或修改代码后重新运行，请按以下步骤操作。

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

#### 网络特征差异分析

| 指标 | 合法/非法倍数比 | 解读 |
|------|----------------|------|
| 出度中心性 | 185.87x | 非法节点对外连接极少，逃避追踪 |
| 聚类系数 | 23.35x | 非法交易更分散，不形成紧密社区 |
| 介数中心性 | 51.87x | 非法节点很少充当网络中介角色 |
| 入度中心性 | 55.47x | 非法节点接收的交易对手较少 |

### 未来展望

针对本次研究的局限性，提出以下改进方向：

#### 1. 改进 GNN 模型

| 改进方向 | 具体方法 | 预期效果 |
|---------|---------|---------|
| 未知节点处理 | 邻域采样时排除未知节点；使用半监督学习方法 | 减少噪声干扰 |
| 验证策略 | 引入时间窗口验证集；添加 Early Stopping | 防止过拟合 |
| 图注意力机制 | 改用 GAT（Graph Attention Network）替代 GraphSAGE | 自适应学习邻居重要性 |
| 特征融合 | 结合网络结构特征与原始特征联合训练 | 提升 GNN 表达能力 |

#### 2. 提升 LightGBM 性能

- **特征工程**：基于网络分析结果构造更多判别性特征（如出度异常检测、聚类系数阈值）
- **集成学习**：结合多个 LightGBM 模型（不同特征子集或不同随机种子）
- **成本敏感学习**：进一步调整类别权重，在 Recall 和 Precision 之间寻找最优平衡

#### 3. 探索其他方法

- **图采样策略**：使用邻居采样减少未知节点影响
- **时间感知模型**：利用时间步信息设计时序感知的图卷积
- **异构图建模**：将交易类型、金额等作为不同类型的节点和边

#### 4. 实际应用建议

- **召回率优先场景**（如初步筛查）：选择 Recall 优化模型，减少漏检
- **精确率优先场景**（如调查取证）：选择 Precision 优化模型，减少误报
- **综合评估**：采用 F1-Score 或 F2-Score（强调召回率）作为主要指标

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
