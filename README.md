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
  - [4.3 启动 Web 仪表盘](#43-启动-web-仪表盘)
- [5. 模块详解](#5-模块详解)
  - [5.1 data_download.py — 数据获取](#51-data_downloadpy--数据获取)
  - [5.2 feature_engineering.py — 特征工程](#52-feature_engineeringpy--特征工程)
  - [5.3 model_lstm.py — LSTM 模型](#53-model_lstmpy--lstm-模型)
  - [5.4 model_gru.py — GRU 模型](#54-model_grupy--gru-模型)
  - [5.5 evaluation.py — 评估与可视化](#55-evaluationpy--评估与可视化)
  - [5.6 main.py — 主流程编排](#56-mainpy--主流程编排)
  - [5.7 web/ — Web 可视化仪表盘](#57-web--web-可视化仪表盘)
- [6. 特征说明](#6-特征说明)
- [7. 模型架构](#7-模型架构)
- [8. 训练超参数](#8-训练超参数)
- [9. 评估指标](#9-评估指标)
- [10. Web 仪表盘使用](#10-web-仪表盘使用)
  - [10.1 评估仪表盘](#101-评估仪表盘)
  - [10.2 未来预测页](#102-未来预测页)
  - [10.3 API 接口](#103-api-接口)
- [11. 运行结果](#11-运行结果)
- [12. 输出文件清单](#12-输出文件清单)
- [13. 如何更换股票](#13-如何更换股票)
- [14. 常见问题](#14-常见问题)
- [15. 跨平台迁移](#15-跨平台迁移)

---

## 1. 项目概述

本项目基于深度学习构建 **LSTM** 与 **GRU** 两种循环神经网络模型，对贵州茅台（600519）股票日线数据进行单步回归预测。利用历史 30 个交易日的 **18 维特征**（5 维基础行情 + 12 维技术指标 + 1 维辅助），预测下一个交易日的**收盘价**，并在相同条件下对两种模型进行横向对比。

### 核心特性

- **数据源**：akshare（免费开源金融数据接口，无需 API Token）
- **特征维度**：18 维（5 OHLCV + 12 技术指标 + CCI）
- **模型对比**：LSTM vs GRU，相同架构仅替换 RNN 单元
- **训练策略**：Early Stopping + 最优权重保存
- **双环境支持**：本地 Conda 开发 + AI Studio GPU 训练，代码完全互通
- **Web 仪表盘**：Flask + ECharts 交互式可视化，含评估对比、残差分析、未来 15 天预测
- **未来预测**：自回归滚动预测，支持任意天数外推

### 完整工作流

```
akshare 下载数据 → 特征工程(18维) → 标准化 + 滑动窗口
                                         ↓
                         ┌───────────────┴───────────────┐
                         ↓                               ↓
                    LSTM 训练                        GRU 训练
                    (34,081 参数)                    (25,569 参数)
                         ↓                               ↓
                         └───────────────┬───────────────┘
                                         ↓
                               评估 + 可视化 + JSON 导出
                                         ↓
                               Flask Web 仪表盘
                              ├─ 评估对比页 (dashboard)
                              └─ 未来预测页 (forecast)
```

---

## 2. 项目结构

```
D:\Stock_price_prediction\
│
├── 股票价格预测_需求文档.md    # 详细需求规格文档
├── README.md                    # 本文件（项目使用文档）
├── requirements.txt             # Python 依赖清单
│
├── main.py                      # 【入口】一键运行全流程（5步）
├── data_download.py             # 模块1：akshare 数据获取
├── feature_engineering.py       # 模块2：技术指标 + 预处理 + 窗口构造
├── model_lstm.py                # 模块3：LSTM 模型定义与训练
├── model_gru.py                 # 模块4：GRU 模型定义与训练
├── evaluation.py                # 模块5：评估指标 + 可视化 + JSON 导出 + 未来预测
│
├── data/                        # 数据目录（运行后生成）
│   ├── 600519_daily.csv         #   原始日线数据（OHLCV）
│   ├── X_train.npy              #   训练集特征 (N, 30, 18)
│   ├── y_train.npy              #   训练集标签 (N,)
│   ├── X_val.npy                #   验证集特征
│   ├── y_val.npy                #   验证集标签
│   ├── X_test.npy               #   测试集特征（标准化）
│   ├── y_test.npy               #   测试集标签（标准化）
│   ├── X_test_unscaled.npy      #   测试集特征（未标准化，备用）
│   ├── y_test_real.npy          #   测试集真实收盘价（反标准化后）
│   ├── dates_test.npy           #   测试集对应日期
│   ├── lstm_test_pred.npy       #   LSTM 预测值（反标准化后）
│   ├── gru_test_pred.npy        #   GRU 预测值（反标准化后）
│   └── scaler_info.npz          #   反标准化参数 (close_mean, close_std)
│
├── models/                      # 模型目录（运行后生成）
│   ├── lstm_best.pt             #   LSTM 最优权重 (~140 KB)
│   ├── gru_best.pt              #   GRU 最优权重 (~106 KB)
│   ├── lstm_history.npz         #   LSTM 训练历史 (train_losses, val_losses, best_epoch)
│   ├── gru_history.npz          #   GRU 训练历史
│   └── scaler.pkl               #   StandardScaler 对象（17 维均值/标准差）
│
├── figures/                     # 图表 + JSON 目录（运行后生成）
│   ├── loss_curves.png          #   训练/验证损失曲线对比
│   ├── predictions.png          #   预测值 vs 真实值对比（最近 200 样本）
│   ├── residuals.png            #   残差分布直方图（LSTM vs GRU）
│   ├── metrics.json             #   评估指标 JSON（供 Web 使用）
│   ├── chart_history.json       #   损失曲线数据 JSON
│   ├── chart_predict.json       #   预测对比数据 JSON
│   ├── chart_residual.json      #   残差分布数据 JSON
│   └── forecast.json            #   未来 15 天预测数据 JSON
│
└── web/                         # Web 可视化（Flask 后端 + ECharts 前端）
    ├── app.py                   #   Flask 应用入口
    └── templates/
        ├── dashboard.html       #   评估仪表盘页面
        └── forecast.html        #   未来预测页面
```

---

## 3. 环境配置

### 3.1 方案A：本地 Conda 环境

**适用场景**：代码开发、逻辑调试、Web 仪表盘展示

#### 环境信息

| 项目 | 值 |
|------|-----|
| 环境名 | `ml_env` |
| 路径 | `D:\ML\ml_env` |
| Python | 3.9 |
| 深度学习框架 | PyTorch 2.8.0 |
| Web 框架 | Flask |

#### 初始化步骤

```bash
# 1. 激活环境
conda activate ml_env

# 2. 安装核心依赖（首次使用）
pip install torch torchvision torchaudio
pip install akshare

# 3. 安装全部依赖
pip install -r requirements.txt

# 4. 验证安装
python -c "import torch; import akshare; import flask; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

#### 主要依赖清单

| 包 | 版本要求 | 用途 |
|----|---------|------|
| torch | ≥2.0.0 | 深度学习框架（LSTM/GRU） |
| akshare | ≥1.14.0 | 金融数据接口 |
| pandas | ≥2.0.0 | 数据处理 |
| numpy | ≥1.26.0 | 数值计算 |
| scikit-learn | ≥1.5.0 | 标准化 + 评估指标 |
| matplotlib | ≥3.9.0 | 静态图表可视化 |
| scipy | ≥1.13.0 | 科学计算 |
| seaborn | ≥0.13.0 | 统计可视化 |
| flask | — | Web 后端服务 |

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
import flask
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

程序将分 5 步执行，且**自动跳过已完成步骤**（幂等设计）：

| 步骤 | 跳过条件 | 覆盖策略 |
|------|----------|----------|
| 数据下载 | `data/600519_daily.csv` 已存在 | 不覆盖 |
| 特征工程 | `data/X_train.npy` 已存在 | 不覆盖 |
| LSTM 训练 | 无跳过 | 每次覆盖旧模型 |
| GRU 训练 | 无跳过 | 每次覆盖旧模型 |
| 评估可视化 | 无跳过 | 每次重新生成 |

### 4.2 分步运行

也可以单独执行每个模块（便于调试）：

```bash
cd D:\Stock_price_prediction

# Step 1: 下载数据（约 10-30 秒，取决于网络）
python data_download.py

# Step 2: 特征工程（约 5 秒）
python feature_engineering.py

# Step 3: 训练 LSTM（CPU 约 15-30 秒，GPU 约 5 秒）
python model_lstm.py

# Step 4: 训练 GRU（CPU 约 10-20 秒，GPU 约 3 秒）
python model_gru.py

# Step 5: 评估 + 可视化 + JSON 导出 + 未来预测（约 10 秒）
python evaluation.py
```

### 4.3 启动 Web 仪表盘

```bash
# 确保已完成 main.py 或分步运行的所有步骤
cd D:\Stock_price_prediction
python web/app.py
```

浏览器访问：
- **评估仪表盘**：http://127.0.0.1:5000/
- **未来预测页**：http://127.0.0.1:5000/forecast
- **API 接口**：http://127.0.0.1:5000/api/metrics 等

> 详细说明见 [第 10 节 Web 仪表盘使用](#10-web-仪表盘使用)

---

## 5. 模块详解

### 5.1 data_download.py — 数据获取

**功能**：通过 akshare 接口拉取股票历史日线数据并保存为 CSV。

**数据流**：
```
东方财富/新浪公开接口
    ↓ akshare.stock_zh_a_hist()
日线数据 (OHLCV)
    ↓ 列名标准化 + 按日期排序
data/600519_daily.csv
```

**核心代码**：
```python
import akshare as ak

df = ak.stock_zh_a_hist(
    symbol="600519",       # 股票代码
    period="daily",        # 日线
    start_date="20200101", # 起始日期
    end_date="20251231",   # 截止日期
    adjust="qfq"           # 前复权
)
```

**函数签名**：
```python
def download_data(symbol: str = "600519",
                  start_date: str = "20200101",
                  end_date: str = "20251231",
                  save_dir: str = "data") -> pd.DataFrame
```

**输出字段**：

| 列名 | 类型 | 说明 |
|------|------|------|
| Date | datetime | 交易日日期 |
| Open | float | 开盘价（前复权） |
| High | float | 最高价（前复权） |
| Low | float | 最低价（前复权） |
| Close | float | 收盘价（前复权） |
| Volume | int | 成交量（手） |

**更换股票**：只需传入不同的 `symbol` 参数。例如 `download_data(symbol="000001")` 下载平安银行数据。

---

### 5.2 feature_engineering.py — 特征工程

**功能**：从 OHLCV 原始数据计算 12 项技术指标，进行 Z-Score 标准化和滑动窗口序列构造，划分训练/验证/测试集。

**处理流程**：

```
CSV 原始数据 (6 列)
    ↓ 计算 9 类技术指标（12 个特征列）
18 维特征矩阵 (含 Date)
    ↓ dropna() 清洗含 NaN 的行
清洗后数据
    ↓ 时序划分 (70% / 15% / 15%)  — 不随机打乱
训练集 / 验证集 / 测试集
    ↓ StandardScaler（仅在训练集拟合 μ, σ）
标准化数据
    ↓ 滑动窗口 (lookback=30, step=1)
X: (样本数, 30, 18)   y: (样本数,)
    ↓ 保存为 .npy
data/X_train.npy … data/y_test.npy
```

**9 个技术指标函数**：

| 函数 | 产出特征 | 参数 |
|------|---------|------|
| `calc_ma()` | MA5, MA10, MA20 | window=5, 10, 20 |
| `calc_macd()` | MACD (DIF−DEA) | fast=12, slow=26, signal=9 |
| `calc_rsi()` | RSI | period=14 |
| `calc_boll()` | BOLL_upper, BOLL_middle, BOLL_lower | window=20, std=2 |
| `calc_atr()` | ATR | period=14 |
| `calc_obv()` | OBV | — |
| `calc_kdj()` | KDJ (K值) | period=9 |
| `calc_wr()` | WR | period=14 |
| `calc_cci()` | CCI | period=20 |

**时序数据划分**：
```
|█████████████████████████████████████████████████████████████|██████████████|██████████████|
              70% 训练集 (~975 天)                             15% 验证集      15% 测试集
                                                                  (~210 天)     (~210 天)
```

**滑动窗口示意**：
```
X[0] = 第 1~30 天  × 18 维特征   →   y[0] = 第 31 天收盘价
X[1] = 第 2~31 天  × 18 维特征   →   y[1] = 第 32 天收盘价
X[2] = 第 3~32 天  × 18 维特征   →   y[2] = 第 33 天收盘价
...
```

**关键设计**：Z-Score 标准化仅在训练集上拟合 scaler，验证集和测试集使用训练集的 μ/σ 进行 transform，防止数据泄露。

**函数签名**：
```python
def process(symbol: str = "600519",
            data_dir: str = "data",
            lookback: int = 30,
            step: int = 1,
            train_ratio: float = 0.7,
            val_ratio: float = 0.15)
```

---

### 5.3 model_lstm.py — LSTM 模型

**模型类**：`LSTMModel`

**网络结构**：

```
Input: (batch, 30, 18)
        │
        ▼
┌──────────────────────────┐
│  LSTM Layer 1            │
│  input_size=18           │
│  hidden_size=64          │
│  batch_first=True        │  参数量: 21,248
│  (默认 1 层)              │
└──────────┬───────────────┘
           │ (batch, 30, 64)
           ▼
┌──────────────────────────┐
│  LSTM Layer 2            │
│  input_size=64           │
│  hidden_size=32          │
│  batch_first=True        │  参数量: 12,544
└──────────┬───────────────┘
           │ (batch, 30, 32)
           │ 取[:, -1, :] → (batch, 32)
           ▼
┌──────────────────────────┐
│  Dropout(p=0.2)          │  防过拟合
└──────────┬───────────────┘
           │ (batch, 32)
           ▼
┌──────────────────────────┐
│  Linear(32, 1)           │  参数量: 33
└──────────┬───────────────┘
           │
           ▼
Output: (batch,)  预测的下一天收盘价（标准化）
```

**总参数量**：**34,081**

**训练配置**：
| 参数 | 值 |
|------|-----|
| 损失函数 | MSELoss |
| 优化器 | Adam (lr=0.001) |
| Batch Size | 32 |
| 最大 Epochs | 100 |
| Early Stopping | patience=10, monitor='val_loss' |
| 权重保存 | 保留 val_loss 最小的 checkpoint |
| 训练集 shuffle | 是（每 epoch 打乱） |

**调用方式**：
```python
from model_lstm import run
history = run()  # 自动加载 data/ 下的预处理数据，返回训练历史
```

---

### 5.4 model_gru.py — GRU 模型

**模型类**：`GRUModel`

**网络结构**（与 LSTM 完全相同，仅将 `nn.LSTM` 替换为 `nn.GRU`）：

```
Input: (batch, 30, 18)
        │
        ▼
┌──────────────────────────┐
│  GRU Layer 1             │
│  input_size=18           │
│  hidden_size=64          │  参数量: 15,936
└──────────┬───────────────┘
           │ (batch, 30, 64)
           ▼
┌──────────────────────────┐
│  GRU Layer 2             │
│  input_size=64           │
│  hidden_size=32          │  参数量: 9,408
└──────────┬───────────────┘
           │ (batch, 30, 32)
           │ 取[:, -1, :] → (batch, 32)
           ▼
┌──────────────────────────┐
│  Dropout(p=0.2)          │
└──────────┬───────────────┘
           │ (batch, 32)
           ▼
┌──────────────────────────┐
│  Linear(32, 1)           │  参数量: 33
└──────────┬───────────────┘
           │
           ▼
Output: (batch,)
```

**总参数量**：**25,569**（比 LSTM 少约 **25%**）

**LSTM vs GRU 门控机制差异**：

| 特性 | LSTM | GRU |
|------|------|-----|
| 门控数量 | 3（遗忘门 + 输入门 + 输出门） | 2（重置门 + 更新门） |
| 细胞状态 | 有独立 cell state c_t | 无，h_t 同时承担 |
| 参数量 | 较多（每个门一套权重） | 较少（少一个门） |
| 训练速度 | 较慢 | 较快 |
| 表达能力 | 理论上更强 | 实践中常与 LSTM 持平 |
| 过拟合风险 | 较高（参数多） | 较低 |

---

### 5.5 evaluation.py — 评估与可视化

**功能**：本模块是整个流程的终点，承担 5 个子任务：

```
            ┌───────────────────────────┐
            │      evaluation.py        │
            └───────────┬───────────────┘
                        │
   ┌────────────────────┼────────────────────┬───────────────────┬──────────────────┐
   ▼                    ▼                     ▼                   ▼                  ▼
1. 模型评估         2. 静态图表         3. JSON 导出        4. 未来预测        5. 控制台输出
   ├─ LSTM 预测       ├─ loss_curves.png   ├─ metrics.json     ├─ 自回归滚动       ├─ 指标对比表
   ├─ GRU 预测        ├─ predictions.png   ├─ chart_history    │   预测 15 天       ├─ 最优 epoch
   ├─ 反标准化         ├─ residuals.png    │   .json           ├─ 跳过周末          └─ 参数量
   └─ 5 项指标         └─ 中文字体          ├─ chart_predict    └─ forecast.json
                                              .json
                                            └─ chart_residual
                                               .json
```

#### 子任务 1：模型评估

1. 加载测试集数据 + 反标准化参数（`scaler_info.npz`）
2. 加载 LSTM / GRU 最优权重（`.pt`）
3. 在测试集上计算预测值，反标准化为真实价格
4. 计算 5 项指标：MSE、RMSE、MAE、MAPE、R²

**5 项评估指标**：

| 指标 | 公式 | 量纲 | 方向 |
|------|------|------|:--:|
| **MSE** | (1/n)Σ(yᵢ − ŷᵢ)² | 价格² | ↓ |
| **RMSE** | √MSE | 元 | ↓ |
| **MAE** | (1/n)Σ\|yᵢ − ŷᵢ\| | 元 | ↓ |
| **MAPE** | (1/n)Σ\|(yᵢ−ŷᵢ)/yᵢ\| × 100% | 百分比 | ↓ |
| **R²** | 1 − SS_res/SS_tot | 无 | ↑ |

#### 子任务 2：静态图表（matplotlib）

| 图表 | 文件 | 内容 |
|------|------|------|
| 损失曲线 | `figures/loss_curves.png` | 双模型 train/val loss 随 epoch 变化，竖线标记最优 epoch |
| 预测对比 | `figures/predictions.png` | 最近 200 样本：真实值 (黑) + LSTM (蓝虚线) + GRU (红虚线) |
| 残差分布 | `figures/residuals.png` | 左右并排直方图，标注 mean/std |

#### 子任务 3：JSON 数据导出

为 Web 仪表盘生成 4 个 JSON 文件（详见 [第 12 节](#12-输出文件清单)）。

#### 子任务 4：未来预测（自回归）

使用 `forecast_future()` 函数进行自回归滚动预测：

```
初始窗口: 最后 30 天真实特征 (30 × 18)
    ↓
Step 1: 模型预测 T+1 天 Close  →  写入窗口 Close 列（索引 3）
Step 2: 窗口滑动 1 天，其余特征沿用   →  预测 T+2 天 Close
...
Step 15: 得到未来 15 天预测值
    ↓
export_forecast_json() → figures/forecast.json
    ↓
Web 前端 forecast.html 读取展示
```

**误差累积说明**：自回归预测每一步都会引入误差，且误差会随步长传播放大。非 Close 特征（技术指标等）沿用上一天数值是近似处理。**预测结果仅供参考，不构成投资建议。**

#### 子任务 5：控制台输出

```text
============================================================
                    评估结果对比
============================================================
指标         LSTM           GRU         较优
--------------------------------------------------
MSE         1126.32        454.75       GRU *
RMSE          33.56         21.33       GRU *
MAE           29.08         16.82       GRU *
MAPE           2.03          1.17       GRU *
R2             0.49          0.80       GRU *
============================================================
```

---

### 5.6 main.py — 主流程编排

**5 步编排流程**：

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Step 1/5   │    │  Step 2/5   │    │  Step 3/5   │    │  Step 4/5   │    │  Step 5/5   │
│  数据下载    │───→│  特征工程    │───→│  LSTM 训练   │───→│  GRU 训练    │───→│  评估可视化  │
│  akshare    │    │  18维 + 窗口 │    │  34K 参数    │    │  26K 参数    │    │  5指标+图表  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       ↓                  ↓                  ↓                  ↓                  ↓
 600519_daily.csv    .npy 数组         lstm_best.pt        gru_best.pt         figures/*
                                                           models/*            data/*
```

**幂等设计**：Step 1 和 Step 2 检查中间产物是否存在，存在则跳过。Step 3-5 每次重新执行以确保使用最新模型。重复运行不会重复下载数据或重复做特征工程。

**错误处理**：
- `KeyboardInterrupt` → 友好退出
- 其他异常 → 打印错误信息并抛出

---

### 5.7 web/ — Web 可视化仪表盘

**文件**：
- `web/app.py` — Flask 后端，提供路由 + API 接口
- `web/templates/dashboard.html` — 评估仪表盘前端（ECharts 图表）
- `web/templates/forecast.html` — 未来预测页前端

#### 路由表

| 路由 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 评估仪表盘主页，读取 `figures/*.json` 渲染 4 张图表 |
| `/forecast` | GET | 未来预测页，读取 `figures/forecast.json` 渲染折线图 + 预测表 |
| `/api/metrics` | GET | 返回 `metrics.json` |
| `/api/history` | GET | 返回 `chart_history.json` |
| `/api/predict` | GET | 返回 `chart_predict.json` |
| `/api/residual` | GET | 返回 `chart_residual.json` |
| `/api/forecast` | GET | 返回 `forecast.json` |

#### 前端图表（dashboard.html）

| 图表 | 类型 | 内容 |
|------|------|------|
| 概览卡片 ×4 | 统计卡片 | LSTM/GRU 参数量、最优轮次 |
| 指标对比表 | 表格 | 5 项指标 + 较优模型标注 |
| 损失曲线 | ECharts 折线图 | 双模型 train/val 四条曲线 + 最优轮次竖线标记 |
| 预测对比 | ECharts 折线图 | 真实值 + LSTM/GRU 预测值 + 缩放滑块 |
| 残差分布 ×2 | ECharts 直方图 | LSTM/GRU 残差分布 + mean=0 标记 |

#### 前端图表（forecast.html）

| 图表 | 类型 | 内容 |
|------|------|------|
| 概览卡片 ×4 | 统计卡片 | 基准收盘价、最后数据日期、双模型 D15 预测值 |
| 历史+预测图 | ECharts 折线图 | 最近 60 天历史 + 未来 15 天预测（虚线），虚线分隔历史/预测区域 |
| 预测数值表 | 表格 | 逐日日期 + LSTM/GRU 预测值 + 差值 |
| 方法说明 | 卡片 | 自回归滚动预测原理 + 风险提示 |

#### 启动方式

```bash
cd D:\Stock_price_prediction
python web/app.py
# Flask 监听 http://127.0.0.1:5000
```

> 必须先运行 `main.py` 或分步运行 `evaluation.py`，确保 `figures/*.json` 已生成。

---

## 6. 特征说明

### 6.1 基础行情数据（5 维）

| 序号 | 特征 | 说明 |
|:----:|------|------|
| 1 | Open | 开盘价（前复权） |
| 2 | High | 最高价（前复权） |
| 3 | Low | 最低价（前复权） |
| 4 | Close | 收盘价（前复权） — 也是预测目标 |
| 5 | Volume | 成交量（手） |

### 6.2 技术指标（12 维）

| 序号 | 特征 | 分类 | 参数 | 含义 |
|:----:|------|------|------|------|
| 6 | MA5 | 趋势 | window=5 | 5 日移动均线 |
| 7 | MA10 | 趋势 | window=10 | 10 日移动均线 |
| 8 | MA20 | 趋势 | window=20 | 20 日移动均线 |
| 9 | MACD | 趋势 | fast=12, slow=26 | DIF−DEA 柱线值 |
| 10 | RSI | 动量 | period=14 | 相对强弱指标，超买/超卖判断 |
| 11 | BOLL_upper | 波动 | window=20, std=2 | 布林带上轨（压力位参考） |
| 12 | BOLL_middle | 波动 | window=20 | 布林带中轨（20日均线） |
| 13 | BOLL_lower | 波动 | window=20, std=2 | 布林带下轨（支撑位参考） |
| 14 | ATR | 波动 | period=14 | 平均真实波幅，衡量波动性 |
| 15 | OBV | 量价 | — | 能量潮，量价配合分析 |
| 16 | KDJ | 动量 | period=9 | 随机指标 K 值 |
| 17 | WR | 动量 | period=14 | 威廉指标，超买超卖 |
| 18 | CCI | 动量 | period=20 | 商品通道指数，突破判断 |

### 6.3 特征列索引（FEATURE_COLUMNS）

```python
FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",     # 0-4
    "MA5", "MA10", "MA20",                         # 5-7
    "MACD", "RSI",                                 # 8-9
    "BOLL_upper", "BOLL_middle", "BOLL_lower",     # 10-12
    "ATR", "OBV", "KDJ", "WR", "CCI",             # 13-17
]
```

**目标列索引**：`target_idx=3`（Close）

### 6.4 序列构造

```
输入 X: shape (样本数, 30, 18)
  │
  │  X[i] = [第 i+1 天的特征, ..., 第 i+30 天的特征]
  │         ← 30 个交易日，每个交易日 18 维特征 →
  │
输出 y: shape (样本数,)
  │
  │  y[i] = 第 i+31 天的收盘价（Close）
  │

示例:
  X[0] = 第 1~30 天 的全部特征  →  y[0] = 第 31 天收盘价
  X[1] = 第 2~31 天 的全部特征  →  y[1] = 第 32 天收盘价
```

---

## 7. 模型架构

### 7.1 共同网络结构

```
Input: (batch, 30, 18)
       │
       ▼
┌─────────────────────────────────────────────┐
│  RNN Layer 1 (LSTM 或 GRU)                  │
│  input_size=18, hidden_size=64              │
│  batch_first=True                            │
│  输出: (batch, 30, 64)  ← 保留完整时序      │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│  RNN Layer 2 (LSTM 或 GRU)                  │
│  input_size=64, hidden_size=32              │
│  batch_first=True                            │
│  输出: (batch, 30, 32)                       │
│  取最后时间步 → (batch, 32)                  │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│  Dropout(p=0.2)                             │
│  输出: (batch, 32)                           │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│  Dense(32 → 1)                              │
│  输出: (batch, 1)                            │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
Output: (batch,)  预测的下一日收盘价（标准化值）
```

### 7.2 LSTM vs GRU 参数量对比

| 层 | LSTM 参数 | GRU 参数 | 差异 |
|----|:--------:|:--------:|:----:|
| RNN Layer 1 | 21,248 | 15,936 | −25% |
| RNN Layer 2 | 12,544 | 9,408 | −25% |
| Dropout | 0 | 0 | — |
| Dense | 33 | 33 | — |
| **总计** | **34,081** | **25,569** | **−25%** |

> LSTM 每层有 4 组权重（input/forget/cell/output），GRU 仅有 3 组（reset/update/new），因此参数量减少约 25%。

---

## 8. 训练超参数

| 超参数 | 值 | 说明 |
|--------|-----|------|
| lookback | 30 | 回看天数（滑动窗口长度） |
| step | 1 | 预测步长（单步） |
| 优化器 | Adam | 自适应学习率优化 |
| 学习率 | 0.001 | 默认学习率 |
| 损失函数 | MSE | 均方误差 |
| Batch Size | 32 | 每批样本数 |
| Epochs | 100 | 最大训练轮数 |
| Early Stopping Patience | 10 | 验证损失连续不降 10 轮后停止 |
| Monitor | val_loss | 早停监控指标 |
| Dropout Rate | 0.2 | 防过拟合正则化 |
| RNN Hidden1 | 64 | 第一层隐藏单元数 |
| RNN Hidden2 | 32 | 第二层隐藏单元数 |
| 数据划分 | 70/15/15 | 训练/验证/测试比例 |
| 标准化 | Z-Score | StandardScaler，仅在训练集拟合 |
| 随机种子 | 无固定 | 允许每次运行探索不同初始化 |

---

## 9. 评估指标

### 9.1 指标公式

#### MSE — 均方误差
$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$
- 对大偏差惩罚更重（平方放大误差）
- 量纲：价格²

#### RMSE — 均方根误差
$$RMSE = \sqrt{MSE}$$
- 量纲与原始价格一致（元），可直观理解为"平均偏差多少元"
- 常用于金融时间序列评估

#### MAE — 平均绝对误差
$$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$
- 所有偏差等权，对异常值不敏感
- 量纲：元

#### MAPE — 平均绝对百分比误差
$$MAPE = \frac{1}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right| \times 100\%$$
- 百分比形式，可跨不同价格区间对比
- ⚠️ 真实值为 0 时无法计算（代码中已用 mask 过滤）

#### R² — 决定系数
$$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$
- 衡量模型对数据方差的解释比例
- 范围：通常 [0, 1]，越接近 1 越好
- 可能为负（模型比均值预测还差）

### 9.2 判断方向

| 指标 | 方向 | 理想值 | 说明 |
|------|:--:|--------|------|
| MSE | ↓ 越小越好 | 0 | 平方惩罚大偏差 |
| RMSE | ↓ 越小越好 | 0 | 与价格同量纲 |
| MAE | ↓ 越小越好 | 0 | 直观平均误差 |
| MAPE | ↓ 越小越好 | 0% | 百分比，跨标的可比 |
| R² | ↑ 越大越好 | 1.0 | 拟合优度 |

---

## 10. Web 仪表盘使用

### 10.1 评估仪表盘

访问 `http://127.0.0.1:5000/` 查看模型评估仪表盘。

**页面结构**：
1. **顶部统计卡片**（4 个）
   - LSTM 参数量：34,081
   - GRU 参数量：25,569
   - LSTM 最优轮次：展示 best_epoch
   - GRU 最优轮次：展示 best_epoch

2. **评估指标对比表**
   - 5 行（MSE / RMSE / MAE / MAPE / R²）
   - 每行标注较优模型（绿色高亮）
   - 最优行有浅绿色底色

3. **训练与验证损失曲线**（全宽）
   - 4 条折线：LSTM 训练/验证 + GRU 训练/验证
   - 训练线半透明，验证线实线
   - 虚线竖线标记最优轮次

4. **预测 vs 真实收盘价**（全宽）
   - 3 条折线：真实值 + LSTM 预测 + GRU 预测
   - 取最后 200 个测试样本
   - 底部滑块支持缩放

5. **残差分布直方图**（左右并排）
   - LSTM 残差分布（蓝色）+ GRU 残差分布（红色）
   - 虚线标记 mean=0
   - 角标显示均值/标准差

**交互功能**：
- 所有图表支持 ECharts 工具箱：保存为图片、区域缩放、重置
- 损失曲线支持内部缩放（dataZoom inside）
- 预测图支持滑块缩放（dataZoom slider）

### 10.2 未来预测页

访问 `http://127.0.0.1:5000/forecast` 查看未来 15 天预测。

**页面结构**：
1. **顶部统计卡片**（4 个）
   - 基准收盘价（最后一天真实值）+ ¥ 符号
   - 最后数据日期
   - LSTM 第 15 天预测值
   - GRU 第 15 天预测值

2. **历史走势 + 未来预测图**
   - 最近 60 天真实历史收盘价（灰色实线）
   - 未来 15 天 LSTM 预测（蓝色虚线 + 圆点）
   - 未来 15 天 GRU 预测（红色虚线 + 菱形点）
   - 金色虚线分隔历史/预测区域
   - 底部滑块缩放

3. **逐日预测数值表**
   - 日期 + LSTM 预测 + GRU 预测 + 差值
   - LSTM 预测值蓝色字体，GRU 预测值红色字体

4. **方法说明卡片**
   - 自回归滚动预测原理（4 步说明）
   - 风险提示（黄色警告框）

### 10.3 API 接口

所有 API 返回 JSON 格式：

```bash
# 评估指标
curl http://127.0.0.1:5000/api/metrics

# 损失曲线数据
curl http://127.0.0.1:5000/api/history

# 预测对比数据（最后 200 个样本）
curl http://127.0.0.1:5000/api/predict

# 残差分布数据
curl http://127.0.0.1:5000/api/residual

# 未来预测数据
curl http://127.0.0.1:5000/api/forecast
```

---

## 11. 运行结果

### 11.1 运行环境

| 项目 | 值 |
|------|-----|
| 操作系统 | Windows |
| Python | 3.9 |
| PyTorch | 2.8.0 |
| 设备 | CPU |
| 总耗时 | 约 51 秒（含数据下载 + 特征工程 + 双模型训练 + 评估） |

### 11.2 训练概况

| 模型 | 最优轮次 | 最优验证损失 | 参数量 | 早停触发于 |
|------|:--------:|:-----------:|:------:|:--------:|
| LSTM | 12 | 0.04266 | 34,081 | Epoch 22 |
| GRU | 13 | 0.02055 | 25,569 | Epoch 23 |

### 11.3 测试集评估结果

| 指标 | LSTM | GRU | 较优模型 |
|------|------:|-----:|:--------:|
| MSE | 1126.32 | **454.75** | **GRU** |
| RMSE | 33.56 | **21.33** | **GRU** |
| MAE | 29.08 | **16.82** | **GRU** |
| MAPE | 2.03% | **1.17%** | **GRU** |
| R² | 0.494 | **0.796** | **GRU** |

### 11.4 结论

1. **GRU 在全部 5 项指标上优于 LSTM**，优势显著
2. GRU 的 MAPE 仅 **1.17%**，即平均预测误差约为实际价格的 1.17%，对于波动较大的股票市场属于优秀水平
3. GRU 的 R² 达到 **0.796**，说明模型能解释约 80% 的价格方差
4. GRU 参数量仅 25,569，为 LSTM（34,081）的 **75%**，训练收敛更快、过拟合风险更低
5. **该任务上 GRU 是更优选择** — 更简单、更快、更准确

> ⚠️ **注意**：以上结果为单次随机初始化训练的结果。由于神经网络的随机性（权重初始化、batch shuffle），每次运行的具体数值会有波动，但 GRU 整体优于 LSTM 的趋势是稳定的。建议多次运行取平均值以获得更可靠的对比结论。

---

## 12. 输出文件清单

### 数据文件 (`data/`)

| 文件 | 格式 | 形状 | 说明 |
|------|------|------|------|
| `600519_daily.csv` | CSV | (~1400 行) | 原始日线数据（OHLCV） |
| `X_train.npy` | NumPy | (~975, 30, 18) | 训练集特征（标准化） |
| `y_train.npy` | NumPy | (~975,) | 训练集标签（标准化） |
| `X_val.npy` | NumPy | (~210, 30, 18) | 验证集特征（标准化） |
| `y_val.npy` | NumPy | (~210,) | 验证集标签（标准化） |
| `X_test.npy` | NumPy | (~210, 30, 18) | 测试集特征（标准化） |
| `y_test.npy` | NumPy | (~210,) | 测试集标签（标准化） |
| `X_test_unscaled.npy` | NumPy | (~210, 30, 18) | 测试集特征（未标准化，备用） |
| `y_test_real.npy` | NumPy | (~210,) | 测试集真实收盘价（反标准化后） |
| `dates_test.npy` | NumPy | (~210,) | 测试集对应日期 |
| `lstm_test_pred.npy` | NumPy | (~210,) | LSTM 预测值（反标准化后） |
| `gru_test_pred.npy` | NumPy | (~210,) | GRU 预测值（反标准化后） |
| `scaler_info.npz` | NumPy | 2 个标量 | close_mean, close_std（反标准化用） |

### 模型文件 (`models/`)

| 文件 | 格式 | 大小 | 说明 |
|------|------|------|------|
| `lstm_best.pt` | PyTorch | ~140 KB | LSTM 最优权重（state_dict） |
| `gru_best.pt` | PyTorch | ~106 KB | GRU 最优权重（state_dict） |
| `lstm_history.npz` | NumPy | ~1 KB | LSTM 训练历史（train_losses, val_losses, best_epoch） |
| `gru_history.npz` | NumPy | ~1 KB | GRU 训练历史 |
| `scaler.pkl` | Pickle | ~1 KB | StandardScaler 对象（18 维均值 + 标准差） |

### 图表文件 (`figures/`)

| 文件 | 格式 | 说明 |
|------|------|------|
| `loss_curves.png` | PNG (150 dpi) | 训练/验证损失曲线 |
| `predictions.png` | PNG (150 dpi) | 预测值 vs 真实值对比 |
| `residuals.png` | PNG (150 dpi) | 残差分布直方图（LSTM vs GRU） |

### JSON 数据文件 (`figures/`)

| 文件 | 说明 | 包含字段 |
|------|------|----------|
| `metrics.json` | 评估指标汇总 | LSTM/GRU 各项指标值、参数量、最优轮次 |
| `chart_history.json` | 损失曲线数据 | lstm_train, lstm_val, gru_train, gru_val, best_epoch |
| `chart_predict.json` | 预测对比数据 | y_true, lstm_pred, gru_pred（各 200 个点） |
| `chart_residual.json` | 残差分布数据 | lstm/gru 残差值列表 + mean/std |
| `forecast.json` | 未来预测数据 | lstm/gru 15 天预测值 + 日期 + 历史 60 天收盘价 |

---

## 13. 如何更换股票

### 最简方式（不改代码）

1. 删除中间产物（让 pipeline 用新数据重新生成）：
   ```bash
   # Windows PowerShell
   Remove-Item data/X_train.npy, data/X_val.npy, data/X_test.npy, data/y_train.npy, data/y_val.npy, data/y_test.npy, data/scaler_info.npz -ErrorAction SilentlyContinue
   Remove-Item models/* -ErrorAction SilentlyContinue
   Remove-Item figures/* -ErrorAction SilentlyContinue
   ```

2. 下载新股票数据（例如平安银行 `000001`），保存为 `data/600519_daily.csv` 覆盖旧文件：
   ```python
   from data_download import download_data
   download_data(symbol="000001")  # 文件名会是 000001_daily.csv
   # 手动重命名为 600519_daily.csv，或改 main.py（见下方）
   ```

3. 运行 `python main.py`

### 正规方式（改代码传参）

修改 `main.py` 中的 `step_download()` 和 `step_features()` 函数，将股票代码作为参数传入：

```python
# main.py 中的改动
def step_download():
    from data_download import download_data
    download_data(symbol="000001")  # 改为目标股票代码

def step_features():
    from feature_engineering import process
    process(symbol="000001")  # 保持一致
```

同样需要先删除中间产物让流水线重新运行。

---

## 14. 常见问题

### Q1: 运行时出现 `FileNotFoundError: 数据文件不存在`

**原因**：尚未下载数据。

**解决**：先运行 `python data_download.py`，或直接 `python main.py`（会自动执行）。

### Q2: 运行时网络错误 / akshare 下载失败

**原因**：akshare 依赖的公开接口（东方财富等）网络不稳定。

**解决**：
- 检查网络连接
- 重试几次（akshare 有重试机制）
- 如持续失败，可手动从东方财富下载 CSV 放入 `data/` 目录，命名为 `600519_daily.csv`

### Q3: CUDA out of memory

**原因**：GPU 显存不足。

**解决**：
- 代码会自动回退到 CPU（`torch.device` 自适应逻辑）
- 本模型参数量小（仅 ~34K），CPU 训练仅需数十秒，完全可行

### Q4: 乱码或 UnicodeEncodeError

**原因**：Windows 控制台 GBK 编码不支持部分 Unicode 字符。

**解决**：
- 不影响运行结果（错误仅影响终端打印）
- 使用 PowerShell / Windows Terminal 而非 CMD
- `chcp 65001` 切换到 UTF-8 编码

### Q5: 如何只重新训练而不重新下载数据？

```bash
python model_lstm.py    # 只重训 LSTM
python model_gru.py     # 只重训 GRU
python evaluation.py    # 只重新评估（不重新训练）
```

### Q6: Web 仪表盘打开后图表空白

**原因**：`figures/*.json` 文件不存在。

**解决**：先运行 `python evaluation.py`（或 `python main.py`），确保 JSON 数据已生成。

### Q7: Flask 启动报端口占用

**原因**：5000 端口已被占用。

**解决**：修改 `web/app.py` 最后一行中的 `port=5000` 为其他端口（如 `port=8080`）。

### Q8: 如何在 AI Studio 上使用 GPU？

创建项目时选择 **PyTorch** 内核，代码中的 `torch.device('cuda' if torch.cuda.is_available() else 'cpu')` 会自动检测并使用 GPU。

### Q9: 预测结果每次运行都不一样？

正常现象。神经网络权重随机初始化 + batch shuffle 导致每次训练结果有波动。如需稳定结果，可固定随机种子：

```python
import torch
import numpy as np
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
```

---

## 15. 跨平台迁移

### 从本地迁移到 AI Studio

```bash
# 本地：确保代码使用相对路径，无硬编码绝对路径

# AI Studio：
# 1. 上传所有 .py 文件 + requirements.txt
# 2. 选择 PyTorch 内核
# 3. 终端执行：
pip install -r requirements.txt
python main.py
```

### 从 AI Studio 下载结果到本地

AI Studio 生成的 `models/`、`figures/`、`data/` 目录可直接下载到本地对应路径。然后在本地运行：

```bash
python evaluation.py    # 重新评估 + 生成 JSON
python web/app.py       # 启动 Web 仪表盘查看结果
```

---

## 许可与引用

- 数据来源：[akshare](https://github.com/akfamily/akshare) — MIT License
- 深度学习框架：[PyTorch](https://pytorch.org/) — BSD License
- 图表库：[ECharts](https://echarts.apache.org/) — Apache 2.0 License
- Web 框架：[Flask](https://flask.palletsprojects.com/) — BSD License
- **本项目仅供学习研究使用，不构成投资建议**

---

> **最后更新**：2025 年 7 月  
> **项目路径**：`D:\Stock_price_prediction\`
