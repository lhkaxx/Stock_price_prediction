"""
feature_engineering.py — 特征工程模块
计算 12 项技术指标 + Z-Score 标准化 + 滑动窗口序列构造 + 数据划分
"""
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler


# ==================== 技术指标计算 ====================

def calc_ma(df: pd.DataFrame, windows: list = [5, 10, 20]) -> pd.DataFrame:
    """计算移动均线 MA5, MA10, MA20"""
    for w in windows:
        df[f"MA{w}"] = df["Close"].rolling(window=w).mean()
    return df


def calc_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """计算 MACD 指标（返回 DIF-DEA 柱线值）"""
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    # 可选扩展 DIF/DEA，这里按需求只输出 MACD 柱
    return df


def calc_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """计算 RSI 相对强弱指标"""
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df["RSI"] = 100.0 - (100.0 / (1.0 + rs))
    return df


def calc_boll(df: pd.DataFrame, window: int = 20, std: int = 2) -> pd.DataFrame:
    """计算布林带 BOLL_upper, BOLL_middle, BOLL_lower"""
    df["BOLL_middle"] = df["Close"].rolling(window=window).mean()
    std_val = df["Close"].rolling(window=window).std()
    df["BOLL_upper"] = df["BOLL_middle"] + std * std_val
    df["BOLL_lower"] = df["BOLL_middle"] - std * std_val
    return df


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """计算 ATR 平均真实波幅"""
    high, low, close = df["High"], df["Low"], df["Close"]
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1/period, adjust=False).mean()
    return df


def calc_obv(df: pd.DataFrame) -> pd.DataFrame:
    """计算 OBV 能量潮指标"""
    obv = [0]
    for i in range(1, len(df)):
        if df["Close"].iloc[i] > df["Close"].iloc[i-1]:
            obv.append(obv[-1] + df["Volume"].iloc[i])
        elif df["Close"].iloc[i] < df["Close"].iloc[i-1]:
            obv.append(obv[-1] - df["Volume"].iloc[i])
        else:
            obv.append(obv[-1])
    df["OBV"] = obv
    return df


def calc_kdj(df: pd.DataFrame, period: int = 9) -> pd.DataFrame:
    """计算 KDJ 随机指标（返回 K 值作为特征）"""
    low_lst = df["Low"].rolling(window=period).min()
    high_lst = df["High"].rolling(window=period).max()
    rsv = (df["Close"] - low_lst) / (high_lst - low_lst + 1e-10) * 100
    df["KDJ"] = rsv.ewm(alpha=1/3, adjust=False).mean()  # K 值
    return df


def calc_wr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """计算 WR 威廉指标"""
    high_n = df["High"].rolling(window=period).max()
    low_n = df["Low"].rolling(window=period).min()
    df["WR"] = (high_n - df["Close"]) / (high_n - low_n + 1e-10) * 100
    return df


def calc_cci(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """计算 CCI 商品通道指数"""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    ma_tp = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df["CCI"] = (tp - ma_tp) / (0.015 * mad + 1e-10)
    return df


# ==================== 特征工程主流程 ====================

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    对原始 OHLCV 数据计算全部 12 项技术指标，返回含 17 维特征的数据框。

    Args:
        df: 含 Date/Open/High/Low/Close/Volume 的 DataFrame

    Returns:
        含全部 17 维特征 + Date 的 DataFrame（含 NaN 行）
    """
    df = df.copy()
    df = calc_ma(df)
    df = calc_macd(df)
    df = calc_rsi(df)
    df = calc_boll(df)
    df = calc_atr(df)
    df = calc_obv(df)
    df = calc_kdj(df)
    df = calc_wr(df)
    df = calc_cci(df)
    return df


FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "MA5", "MA10", "MA20",
    "MACD", "RSI",
    "BOLL_upper", "BOLL_middle", "BOLL_lower",
    "ATR", "OBV", "KDJ", "WR", "CCI",
]


def create_sequences(data: np.ndarray, target_idx: int = 3, lookback: int = 30, step: int = 1):
    """
    滑动窗口构造监督学习样本。

    Args:
        data: shape (N, features) 的标准化数组
        target_idx: 目标列索引（Close=3）
        lookback: 回看天数
        step: 预测步长

    Returns:
        X: shape (samples, lookback, features)
        y: shape (samples,)
    """
    X, y = [], []
    for i in range(len(data) - lookback - step + 1):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback + step - 1, target_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def process(symbol: str = "600519",
            data_dir: str = "data",
            lookback: int = 30,
            step: int = 1,
            train_ratio: float = 0.7,
            val_ratio: float = 0.15):
    """
    完整数据预处理流程：
    1. 加载 CSV → 2. 计算技术指标 → 3. 清洗 NaN →
    4. 时序划分 → 5. Z-Score 标准化 → 6. 滑动窗口 → 7. 保存

    Returns:
        dict: 包含各数据集及 scaler 信息
    """
    # 1. 加载数据
    csv_path = os.path.join(data_dir, f"{symbol}_daily.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"数据文件不存在: {csv_path}，请先运行 data_download.py")

    df = pd.read_csv(csv_path, parse_dates=["Date"])
    print(f"[feature_engineering] 已加载 {len(df)} 条原始数据")

    # 2. 计算技术指标
    df = build_features(df)
    df = df.dropna().reset_index(drop=True)
    print(f"[feature_engineering] 技术指标计算完成，清洗 NaN 后剩余 {len(df)} 条")

    # 3. 提取特征矩阵
    feature_data = df[FEATURE_COLUMNS].values.astype(np.float64)

    # 4. 时序划分（不打乱顺序）
    n = len(feature_data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_raw = feature_data[:train_end]
    val_raw = feature_data[train_end:val_end]
    test_raw = feature_data[val_end:]

    print(f"[feature_engineering] 数据划分 — 训练:{len(train_raw)} 验证:{len(val_raw)} 测试:{len(test_raw)}")

    # 5. Z-Score 标准化（仅在训练集拟合）
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_raw)
    val_scaled = scaler.transform(val_raw)
    test_scaled = scaler.transform(test_raw)

    # 保存 scaler
    os.makedirs("models", exist_ok=True)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # 6. 滑动窗口
    X_train, y_train = create_sequences(train_scaled, target_idx=3, lookback=lookback, step=step)
    X_val, y_val = create_sequences(val_scaled, target_idx=3, lookback=lookback, step=step)
    X_test, y_test = create_sequences(test_scaled, target_idx=3, lookback=lookback, step=step)

    print(f"[feature_engineering] 序列构造完成 — 训练样本:{X_train.shape[0]} 验证:{X_val.shape[0]} 测试:{X_test.shape[0]}")
    print(f"[feature_engineering] 输入形状: {X_train.shape[1:]} (lookback={lookback}, features={len(FEATURE_COLUMNS)})")

    # 7. 保存为 npy
    os.makedirs(data_dir, exist_ok=True)
    np.save(os.path.join(data_dir, "X_train.npy"), X_train)
    np.save(os.path.join(data_dir, "y_train.npy"), y_train)
    np.save(os.path.join(data_dir, "X_val.npy"), X_val)
    np.save(os.path.join(data_dir, "y_val.npy"), y_val)
    np.save(os.path.join(data_dir, "X_test.npy"), X_test)
    np.save(os.path.join(data_dir, "y_test.npy"), y_test)

    # 额外保存反标准化用信息
    close_mean = scaler.mean_[3]
    close_std = scaler.scale_[3]
    np.savez(os.path.join(data_dir, "scaler_info.npz"),
             close_mean=close_mean, close_std=close_std)

    print(f"[feature_engineering] 数据已保存至 {data_dir}/")
    print(f"[feature_engineering] Close 均值={close_mean:.4f} 标准差={close_std:.4f}")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "close_mean": close_mean, "close_std": close_std,
    }


if __name__ == "__main__":
    process()
