# 股票价格预测 — LSTM vs GRU 项目文档

> **课题方向**：时序预测类（深度学习）  
> **预测标的**：贵州茅台（600519）  
> **时间范围**：2020-01-01 至 2025-12-31  
> **框架**：PyTorch  
> **项目路径**：`D:\Stock_price_prediction\`

---

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 项目结构](#2-项目结构)
- [3. 环境配置](#3-环境配置)
  - [3.1 方案A：本地 Conda 环境](#31-方案a本地-conda-环境)
  - [3.2 方案B：百度 AI Studio](#32-方案b百度-ai-studio)
  - [3.3 环境验证](#33-环境验证)
- [4. 快速开始](#4-快速开始)
  - [4.1 一键运行](#41-一键运行)
  - [4.2 分步运行](#42-分步运行)
- [5. 模块详解](#5-模块详解)
  - [5.1 data_download.py — 数据获取](#51-data_downloadpy--数据获取)
  - [5.2 feature_engineering.py — 特征工程](#52-feature_engineeringpy--特征工程)
  - [5.3 model_lstm.py — LSTM 模型](#53-model_lstmpy--lstm-模型)
  - [5.4 model_gru.py — GRU 模型](#54-model_grupy--gru-模型)
  - [5.5 evaluation.py — 评估与可视化](#55-evaluationpy--评估与可视化)
  - [5.6 main.py — 主流程编排](#56-mainpy--主流程编排)
- [6. 特征说明](#6-特征说明)
- [7. 模型架构](#7-模型架构)
- [8. 训练超参数](#8-训练超参数)
- [9. 评估指标](#9-评估指标)
- [10. 运行结果](#10-运行结果)
- [11. 输出文件清单](#11-输出文件清单)
- [12. 常见问题](#12-常见问题)
- [13. 跨平台迁移](#13-跨平台迁移)

---

## 1. 项目概述

本项目基于深度学习构建 **LSTM** 与 **GRU** 两种循环神经网络模型，对贵州茅台（600519）股票日线数据进行单步回归预测。利用历史 30 个交易日的 **17 维特征**（5 维基础行情 + 12 维技术指标），预测下一个交易日的**收盘价**，并在相同条件下对两种模型进行横向对比。

### 核心特性

- **数据源**：akshare（免费开源金融数据接口，无需 API Token）
- **特征维度**：17 维（5 基础 + 12 技术指标）
- **模型对比**：LSTM vs GRU，相同架构仅替换 RNN 单元
- **训练策略**：Early Stopping + 最优权重保存
- **双环境支持**：本地 Conda 开发 + AI Studio GPU 训练，代码完全互通

---

## 2. 项目结构

```
D:\Stock_price_prediction\
│
├── 股票价格预测_需求文档.md    # 详细需求规格文档
├── README.md                    # 本文件（项目使用文档）
├── requirements.txt             # Python 依赖清单
│
├── main.py                      # 【入口】一键运行全流程
├── data_download.py             # 模块1：akshare 数据获取
├── feature_engineering.py       # 模块2：技术指标 + 预处理
├── model_lstm.py                # 模块3：LSTM 模型定义与训练
├── model_gru.py                 # 模块4：GRU 模型定义与训练
├── evaluation.py                # 模块5：评估指标 + 可视化
│
├── data/                        # 数据目录（运行后生成）
│   ├── 600519_daily.csv         #   原始日线数据
│   ├── X_train.npy              #   训练集特征 (n_samples, 30, 17)
│   ├── y_train.npy              #   训练集标签 (n_samples,)
│   ├── X_val.npy                #   验证集特征
│   ├── y_val.npy                #   验证集标签
│   ├── X_test.npy               #   测试集特征
│   ├── y_test.npy               #   测试集标签
│   └── scaler_info.npz          #   反标准化参数
│
├── models/                      # 模型目录（运行后生成）
│   ├── lstm_best.pt             #   LSTM 最优权重
│   ├── gru_best.pt              #   GRU 最优权重
│   ├── lstm_history.npz         #   LSTM 训练历史
│   ├── gru_history.npz          #   GRU 训练历史
│   └── scaler.pkl               #   StandardScaler 对象
│
└── figures/                     # 图表目录（运行后生成）
    ├── loss_curves.png          #   训练损失曲线对比
    ├── predictions.png          #   预测值 vs 真实值对比
    └── residuals.png            #   残差分布直方图
```

---

## 3. 环境配置

### 3.1 方案A：本地 Conda 环境

**适用场景**：代码开发、逻辑调试、小规模测试

#### 环境信息

| 项目 | 值 |
|------|-----|
| 环境名 | `ml_env` |
| 路径 | `D:\ML\ml_env` |
| Python | 3.9 |
| 深度学习框架 | PyTorch 2.8.0 |

#### 初始化步骤

```bash
# 1. 激活环境
conda activate ml_env

# 2. 安装核心依赖（首次使用）
pip install torch torchvision torchaudio
pip install akshare

# 3. 安装其他依赖（如需要）
pip install -r requirements.txt

# 4. 验证安装
python -c "import torch; import akshare; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

#### 已安装包清单

| 包 | 版本 | 用途 |
|----|------|------|
| torch | 2.8.0 | 深度学习框架（LSTM/GRU） |
| akshare | 1.18.64 | 金融数据接口 |
| pandas | 2.3.3 | 数据处理 |
| numpy | 1.26.4 | 数值计算 |
| scikit-learn | 1.5.1 | 标准化 + 评估指标 |
| matplotlib | 3.9.4 | 可视化 |
| scipy | 1.13.1 | 科学计算 |
| seaborn | 0.13.2 | 统计可视化 |

### 3.2 方案B：百度 AI Studio

**适用场景**：完整 100 epoch 训练、GPU 加速、作业交付

#### 使用流程

1. 打开 [aistudio.baidu.com](https://aistudio.baidu.com)
2. 创建项目 → 选择内核为 **PyTorch**（⚠️ 不是 PaddlePaddle）
3. 上传以下文件到项目根目录：
   ```
   main.py
   data_download.py
   feature_engineering.py
   model_lstm.py
   model_gru.py
   evaluation.py
   requirements.txt
   ```
4. 打开终端，执行：
   ```bash
   pip install -r requirements.txt
   python main.py
   ```

#### 双环境代码一致性保障

| 措施 | 说明 |
|------|------|
| 框架统一 | 使用 PyTorch，不在代码中引入 PaddlePaddle API |
| 相对路径 | 所有文件读写使用相对路径（如 `data/`, `models/`） |
| 设备自适应 | `torch.device('cuda' if torch.cuda.is_available() else 'cpu')` |
| requirements.txt | 固定依赖版本，两边安装一致 |

### 3.3 环境验证

运行以下命令确认环境就绪：

```bash
python -c "
import torch
import akshare
import pandas
import numpy
import sklearn
import matplotlib
print('全部依赖导入成功')
print(f'PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}')
"
```

---

## 4. 快速开始

### 4.1 一键运行

```bash
# 本地
conda activate ml_env
cd D:\Stock_price_prediction
python main.py
```

程序将**自动跳过已完成步骤**（幂等设计），重复运行不会浪费算力：

| 步骤 | 跳过条件 |
|------|----------|
| 数据下载 | `data/600519_daily.csv` 已存在 |
| 特征工程 | `data/X_train.npy` 已存在 |
| LSTM 训练 | 每次重新训练（覆盖旧模型） |
| GRU 训练 | 每次重新训练（覆盖旧模型） |
| 评估可视化 | 每次重新生成图表 |

### 4.2 分步运行

也可以单独执行每个模块（便于调试）：

```bash
cd D:\Stock_price_prediction

# Step 1: 下载数据（约 10-30 秒）
python data_download.py

# Step 2: 特征工程（约 5 秒）
python feature_engineering.py

# Step 3: 训练 LSTM（CPU 约 15-30 秒）
python model_lstm.py

# Step 4: 训练 GRU（CPU 约 10-20 秒）
python model_gru.py

# Step 5: 评估与可视化（约 5 秒）
python evaluation.py
```

---

## 5. 模块详解

### 5.1 data_download.py — 数据获取

**功能**：通过 akshare 接口拉取贵州茅台历史日线数据并保存为 CSV。

**数据流**：
```
东方财富/新浪公开接口
    ↓ akshare.stock_zh_a_hist()
600519 日线数据 (2020-01 至 2025-12)
    ↓ 列名标准化 + 排序
data/600519_daily.csv
```

**核心代码**：
```python
import akshare as ak

df = ak.stock_zh_a_hist(
    symbol="600519",
    period="daily",
    start_date="20200101",
    end_date="20251231",
    adjust="qfq"  # 前复权
)
```

**输出字段**：

| 列名 | 类型 | 说明 |
|------|------|------|
| Date | datetime | 交易日日期 |
| Open | float | 开盘价 |
| High | float | 最高价 |
| Low | float | 最低价 |
| Close | float | 收盘价（前复权） |
| Volume | int | 成交量（手） |

---

### 5.2 feature_engineering.py — 特征工程

**功能**：从 OHLCV 原始数据计算 12 项技术指标，进行 Z-Score 标准化和滑动窗口序列构造，划分训练/验证/测试集。

**处理流程**：

```
CSV 原始数据 (6列)
    ↓ 计算技术指标
17 维特征矩阵
    ↓ dropna() 清洗 NaN
清洗后数据
    ↓ 时序划分 (70/15/15)
训练集 / 验证集 / 测试集
    ↓ StandardScaler (拟合训练集)
标准化数据
    ↓ 滑动窗口 (lookback=30, step=1)
X: (样本数, 30, 17)  y: (样本数,)
    ↓ 保存为 .npy
data/X_train.npy ... data/y_test.npy
```

**12 项技术指标计算详情**：

| 函数 | 指标 | 参数 | 计算方式 |
|------|------|------|----------|
| `calc_ma()` | MA5, MA10, MA20 | window=5/10/20 | `Close.rolling(w).mean()` |
| `calc_macd()` | MACD | fast=12, slow=26 | EMA-fast − EMA-slow |
| `calc_rsi()` | RSI | period=14 | 100 − 100/(1 + avg_gain/avg_loss) |
| `calc_boll()` | BOLL | window=20, std=2 | middle ± 2×std |
| `calc_atr()` | ATR | period=14 | TR 的 EMA 平滑 |
| `calc_obv()` | OBV | — | 涨+量 / 跌−量 累积 |
| `calc_kdj()` | KDJ (K值) | period=9 | RSV 的 EMA 平滑 |
| `calc_wr()` | WR | period=14 | (high_n − close) / (high_n − low_n) |
| `calc_cci()` | CCI | period=20 | (tp − ma_tp) / (0.015 × mad) |

**数据划分比例**：
```
|██████████████████████████████████████████████████████████████████████|███████████████|███████████████|
                       70% 训练集                                           15% 验证集       15% 测试集
```

---

### 5.3 model_lstm.py — LSTM 模型

**模型类**：`LSTMModel`

**网络结构**：

| 层 | 类型 | 输入 → 输出 | 参数量 |
|----|------|-------------|--------|
| lstm1 | LSTM | (batch, 30, 17) → (batch, 30, 64) | 21,248 |
| lstm2 | LSTM | (batch, 30, 64) → (batch, 30, 32) | 12,544 |
| — | 取最后时间步 | (batch, 30, 32) → (batch, 32) | 0 |
| dropout | Dropout(0.2) | (batch, 32) → (batch, 32) | 0 |
| fc | Linear | (batch, 32) → (batch, 1) | 33 |

**总参数量**：34,081

**训练配置**：
- 损失函数：MSELoss
- 优化器：Adam (lr=0.001)
- Batch Size：32
- Early Stopping：patience=10, monitor='val_loss'
- 权重保存：保留 val_loss 最小的 checkpoint

**调用方式**：
```python
from model_lstm import run
history = run()  # 自动加载 data/ 下的预处理数据
```

---

### 5.4 model_gru.py — GRU 模型

**模型类**：`GRUModel`

**网络结构**（与 LSTM 完全相同，仅 RNN 单元替换为 GRU）：

| 层 | 类型 | 输入 → 输出 | 参数量 |
|----|------|-------------|--------|
| gru1 | GRU | (batch, 30, 17) → (batch, 30, 64) | 15,936 |
| gru2 | GRU | (batch, 30, 64) → (batch, 30, 32) | 9,408 |
| — | 取最后时间步 | (batch, 30, 32) → (batch, 32) | 0 |
| dropout | Dropout(0.2) | (batch, 32) → (batch, 32) | 0 |
| fc | Linear | (batch, 32) → (batch, 1) | 33 |

**总参数量**：25,569（比 LSTM 少约 25%）

> LSTM 有 3 个门（遗忘门、输入门、输出门），GRU 只有 2 个门（重置门、更新门），因此 GRU 参数量更少、训练更快。

---

### 5.5 evaluation.py — 评估与可视化

**功能**：
1. 加载测试集数据和两个模型的最优权重
2. 计算预测值并反标准化为真实价格
3. 输出 5 项评估指标对比表
4. 生成 3 张可视化图表

**5 项评估指标**：

| 指标 | 公式 | 意义 |
|------|------|------|
| MSE | (1/n)Σ(y_i - ŷ_i)² | 均方误差，放大惩罚大偏差 |
| RMSE | √MSE | 量纲与原始价格一致 |
| MAE | (1/n)Σ|y_i - ŷ_i| | 平均绝对误差 |
| MAPE | (1/n)Σ|(y_i-ŷ_i)/y_i| × 100% | 百分比误差，可跨标的比较 |
| R² | 1 − SS_res/SS_tot | 拟合优度，越接近 1 越好 |

**3 张可视化图表**：

| 图表 | 文件 | 内容 |
|------|------|------|
| 训练损失曲线 | `figures/loss_curves.png` | LSTM vs GRU 的 train/val loss 随 epoch 变化，标记最优 epoch |
| 预测 vs 真实 | `figures/predictions.png` | 最近 200 个样本的三条曲线（真实值、LSTM 预测、GRU 预测） |
| 残差分布 | `figures/residuals.png` | 两个模型的残差 (y_true − y_pred) 直方图，含均值/标准差 |

---

### 5.6 main.py — 主流程编排

**5 步流程**：

```
 Step 1/5          Step 2/5          Step 3/5          Step 4/5          Step 5/5
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│ 数据下载  │ ──→ │ 特征工程  │ ──→ │ LSTM训练  │ ──→ │ GRU训练   │ ──→ │ 评估可视化 │
│ akshare  │     │ 17维特征  │     │ 34K参数   │     │ 26K参数   │     │ 5指标+3图 │
└──────────┘     └──────────┘     └──────────┘     └──────────┘     └──────────┘
    CSV              .npy             .pt               .pt             .png
```

**幂等设计**：每一步执行前检查中间产物是否已存在，存在则跳过，支持断点续跑。

---

## 6. 特征说明

### 6.1 基础行情数据（5 维）

| 序号 | 特征 | 说明 |
|:----:|------|------|
| 1 | Open | 开盘价 |
| 2 | High | 最高价 |
| 3 | Low | 最低价 |
| 4 | Close | 收盘价（前复权） |
| 5 | Volume | 成交量 |

### 6.2 技术指标（12 维）

| 序号 | 特征 | 类型 | 参数 |
|:----:|------|------|------|
| 6 | MA5 | 趋势 | window=5 |
| 7 | MA10 | 趋势 | window=10 |
| 8 | MA20 | 趋势 | window=20 |
| 9 | MACD | 趋势 | fast=12, slow=26, signal=9 |
| 10 | RSI | 动量 | period=14 |
| 11 | BOLL_upper | 波动 | window=20, std=2 |
| 12 | BOLL_middle | 波动 | window=20 |
| 13 | BOLL_lower | 波动 | window=20 |
| 14 | ATR | 波动 | period=14 |
| 15 | OBV | 量价 | — |
| 16 | KDJ | 动量 | period=9 |
| 17 | WR | 动量 | period=14 |
| 18 | CCI | 动量 | period=20 |

### 6.3 序列构造

```
输入 X: [t-29, t-28, ..., t-1, t]    ← 过去 30 个交易日 × 17 维特征
输出 y: Close_{t+1}                   ← 下一交易日收盘价（标量）

示例：
  X[0] = 第1~30天的全部特征  →  y[0] = 第31天收盘价
  X[1] = 第2~31天的全部特征  →  y[1] = 第32天收盘价
  ...
```

### 6.4 数据标准化

采用 **Z-Score 标准化**：

$$\hat{x} = \frac{x - \mu}{\sigma}$$

- 仅在训练集上计算均值 μ 和标准差 σ
- 验证集、测试集复用训练集的 μ 和 σ，防止数据泄露
- 最终评估时将预测值反标准化为真实价格量纲

---

## 7. 模型架构

### 7.1 共同网络结构

```
Input: (batch, 30, 17)
       │
       ▼
┌─────────────────────┐
│  LSTM / GRU         │  ← 第一层 RNN (64 单元)
│  return_sequences=True│   保留完整时序输出
└─────────┬───────────┘
          │ (batch, 30, 64)
          ▼
┌─────────────────────┐
│  LSTM / GRU         │  ← 第二层 RNN (32 单元)
│  return_sequences=False│  只输出最后时间步
└─────────┬───────────┘
          │ (batch, 32)
          ▼
┌─────────────────────┐
│  Dropout(0.2)       │  ← 正则化防过拟合
└─────────┬───────────┘
          │ (batch, 32)
          ▼
┌─────────────────────┐
│  Dense(1)           │  ← 全连接输出层
└─────────┬───────────┘
          │
          ▼
Output: (batch, 1)   ← 预测的下一日收盘价
```

### 7.2 LSTM vs GRU 对比

| 对比维度 | LSTM | GRU |
|----------|------|-----|
| 门控机制 | 遗忘门 + 输入门 + 输出门 | 重置门 + 更新门 |
| 参数量 | 34,081 | 25,569 |
| 训练速度 | 较慢 | 较快 |
| 表达能力 | 更强（3 门控制） | 简洁（2 门控制） |
| 适用场景 | 长序列、复杂依赖 | 中等长度序列、数据量有限 |

---

## 8. 训练超参数

| 超参数 | 值 | 说明 |
|--------|-----|------|
| lookback | 30 | 回看天数 |
| step | 1 | 预测步长（单步） |
| 优化器 | Adam | 自适应学习率优化 |
| 学习率 | 0.001 | 默认学习率 |
| 损失函数 | MSE | 均方误差 |
| Batch Size | 32 | 每批样本数 |
| Epochs | 100 | 最大训练轮数 |
| Early Stopping Patience | 10 | 验证损失不再下降后等待轮数 |
| Monitor | val_loss | 早停监控指标 |
| Dropout Rate | 0.2 | 防过拟合 |
| RNN Hidden1 | 64 | 第一层隐藏单元数 |
| RNN Hidden2 | 32 | 第二层隐藏单元数 |

---

## 9. 评估指标

### 9.1 指标公式与意义

#### MSE — Mean Squared Error（均方误差）

$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

- **特点**：对大偏差惩罚更重（平方放大误差）
- **量纲**：价格²
- **越小越好**

#### RMSE — Root Mean Squared Error（均方根误差）

$$RMSE = \sqrt{MSE}$$

- **特点**：量纲与原始价格一致，可直观理解为"平均偏差多少钱"
- **量纲**：元
- **越小越好**

#### MAE — Mean Absolute Error（平均绝对误差）

$$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

- **特点**：对异常值不敏感，所有偏差等权
- **量纲**：元
- **越小越好**

#### MAPE — Mean Absolute Percentage Error（平均绝对百分比误差）

$$MAPE = \frac{1}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right| \times 100\%$$

- **特点**：百分比形式，可跨不同价格区间的标的对比
- **量纲**：无（百分比）
- **越小越好**
- ⚠️ 当真实值为 0 时无法计算（本项目中已处理）

#### R² — Coefficient of Determination（决定系数）

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

- **特点**：衡量模型对数据方差的解释比例
- **范围**：通常 [0, 1]，越接近 1 越好（可能为负，表示模型比均值预测还差）
- **越大越好**

### 9.2 指标判断方向汇总

| 指标 | 方向 | 理想值 |
|------|:--:|--------|
| MSE | ↓ 越小越好 | 0 |
| RMSE | ↓ 越小越好 | 0 |
| MAE | ↓ 越小越好 | 0 |
| MAPE | ↓ 越小越好 | 0% |
| R² | ↑ 越大越好 | 1.0 |

---

## 10. 运行结果

### 10.1 最近一次运行摘要

```
运行环境：CPU (本地 ml_env)
总耗时：约 51 秒
Python：3.9
PyTorch：2.8.0
```

### 10.2 训练概况

| 模型 | 最优 Epoch | 最优 Val Loss | 参数量 | 触发早停于 |
|------|:----------:|:-------------:|:------:|:----------:|
| LSTM | 12 | 0.04266 | 34,081 | Epoch 22 |
| GRU | 13 | 0.02055 | 25,569 | Epoch 23 |

### 10.3 测试集评估结果

| 指标 | LSTM | GRU | 较优模型 |
|------|------:|-----:|:--------:|
| MSE | 1126.32 | **454.75** | **GRU** |
| RMSE | 33.56 | **21.33** | **GRU** |
| MAE | 29.08 | **16.82** | **GRU** |
| MAPE | 2.03% | **1.17%** | **GRU** |
| R² | 0.494 | **0.796** | **GRU** |

### 10.4 结论

1. **GRU 在全部五项指标上优于 LSTM**，且优势显著
2. GRU 的 MAPE 仅 1.17%，即平均预测误差约为实际价格的 1.17%，对于波动较大的股票市场属于可接受范围
3. GRU 的 R² 达到 0.796，说明模型能解释约 80% 的价格方差
4. GRU 参数量仅为 LSTM 的 75%，训练收敛更快
5. **该任务上 GRU 是更优选择** — 更简单、更快、更准确

> ⚠️ 注意：以上结果为单次随机初始化训练的结果。由于神经网络的随机性（权重初始化、batch shuffle），每次运行的具体数值会有波动，但 GRU 整体优于 LSTM 的趋势是稳定的。

---

## 11. 输出文件清单

### 数据文件 (`data/`)

| 文件 | 格式 | 形状 | 说明 |
|------|------|------|------|
| `600519_daily.csv` | CSV | — | 原始日线数据 |
| `X_train.npy` | NumPy | (~975, 30, 17) | 训练集特征 |
| `y_train.npy` | NumPy | (~975,) | 训练集标签 |
| `X_val.npy` | NumPy | (~210, 30, 17) | 验证集特征 |
| `y_val.npy` | NumPy | (~210,) | 验证集标签 |
| `X_test.npy` | NumPy | (~210, 30, 17) | 测试集特征 |
| `y_test.npy` | NumPy | (~210,) | 测试集标签 |
| `scaler_info.npz` | NumPy | — | close_mean, close_std |

### 模型文件 (`models/`)

| 文件 | 格式 | 大小 | 说明 |
|------|------|------|------|
| `lstm_best.pt` | PyTorch | ~140 KB | LSTM 最优权重 |
| `gru_best.pt` | PyTorch | ~106 KB | GRU 最优权重 |
| `lstm_history.npz` | NumPy | ~1 KB | LSTM 训练历史 |
| `gru_history.npz` | NumPy | ~1 KB | GRU 训练历史 |
| `scaler.pkl` | Pickle | ~1 KB | StandardScaler 对象 |

### 图表文件 (`figures/`)

| 文件 | 格式 | 说明 |
|------|------|------|
| `loss_curves.png` | PNG (150 dpi) | 训练/验证损失曲线 |
| `predictions.png` | PNG (150 dpi) | 预测值 vs 真实值 |
| `residuals.png` | PNG (150 dpi) | 残差分布直方图 |

---

## 12. 常见问题

### Q1: 运行时出现 `FileNotFoundError: 数据文件不存在`

**原因**：尚未下载数据。

**解决**：先运行 `python data_download.py`，或直接 `python main.py`（会自动执行）。

### Q2: 运行时网络错误 / akshare 下载失败

**原因**：akshare 依赖的公开接口（东方财富等）网络不稳定。

**解决**：
- 检查网络连接
- 重试几次（大文件已缓存，断点续传）
- 如持续失败可手动从东方财富下载 CSV 放入 `data/` 目录

### Q3: CUDA out of memory

**原因**：GPU 显存不足。

**解决**：
- 代码会自动使用 CPU（见 `device` 自适应逻辑）
- 本模型参数量小（~34K），CPU 训练仅需数十秒

### Q4: 乱码或 UnicodeEncodeError

**原因**：Windows 控制台 GBK 编码不支持部分 Unicode 字符。

**解决**：
- 不影响运行结果（错误仅在最后打印环节）
- 在文件开头加 `# -*- coding: utf-8 -*-`
- 或使用 PowerShell / Windows Terminal 而非 CMD
- `chcp 65001` 切换到 UTF-8 编码

### Q5: 如何只重新训练而不重新下载数据？

直接运行训练模块：

```bash
python model_lstm.py    # 只重训 LSTM
python model_gru.py     # 只重训 GRU
python evaluation.py     # 只重新评估（不重新训练）
```

### Q6: 如何在 AI Studio 上使用 GPU？

创建项目时选择 **PyTorch** 内核，代码中的 `torch.device('cuda' if torch.cuda.is_available() else 'cpu')` 会自动检测并使用 GPU。

### Q7: 如何更换预测其他股票？

修改函数参数即可，例如预测平安银行：

```python
# 在 main.py 或单独调用时传入
from data_download import download_data
df = download_data(symbol="000001")  # 平安银行
```

---

## 13. 跨平台迁移

### 从本地迁移到 AI Studio

```bash
# 本地：确保代码使用相对路径，无硬编码

# AI Studio：
# 1. 上传所有 .py 文件 + requirements.txt
# 2. 选择 PyTorch 内核
# 3. 终端执行：
pip install -r requirements.txt
python main.py
```

### 从 AI Studio 下载结果到本地

AI Studio 生成的 `models/`、`figures/`、`data/` 目录可直接下载到本地对应路径，在本地运行 `python evaluation.py` 即可复现图表。

---

## 许可与引用

- 数据来源：[akshare](https://github.com/akfamily/akshare) — MIT License
- 深度学习框架：[PyTorch](https://pytorch.org/) — BSD License
- 本项目仅供学习研究使用，不构成投资建议

---

> **最后更新**：2025 年 7 月  
> **作者**：—  
> **项目路径**：`D:\Stock_price_prediction\`
