"""
evaluation.py — 评估指标计算 + 对比可视化
对 LSTM 和 GRU 模型在测试集上计算 MSE/RMSE/MAE/MAPE/R²，生成对比图表
"""
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")  # 非交互后端，兼容无 GUI 环境
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 中文字体设置
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from model_lstm import LSTMModel
from model_gru import GRUModel


# ==================== 指标计算 ====================

def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """计算全部 5 项评估指标"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    # MAPE：避免除零
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    r2 = r2_score(y_true, y_pred)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


def inverse_scale(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Z-Score 反标准化"""
    return values * std + mean


# ==================== 可视化 ====================

def plot_loss_curves(lstm_history: dict, gru_history: dict, save_dir: str = "figures"):
    """绘制训练损失曲线对比"""
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    epochs_lstm = range(1, len(lstm_history["train_losses"]) + 1)
    epochs_gru = range(1, len(gru_history["train_losses"]) + 1)

    ax.plot(epochs_lstm, lstm_history["train_losses"], "b-", alpha=0.3, label="LSTM Train")
    ax.plot(epochs_lstm, lstm_history["val_losses"], "b-", label="LSTM Val")
    ax.plot(epochs_gru, gru_history["train_losses"], "r-", alpha=0.3, label="GRU Train")
    ax.plot(epochs_gru, gru_history["val_losses"], "r-", label="GRU Val")

    # 标记最优 epoch
    ax.axvline(lstm_history["best_epoch"], color="b", linestyle="--", alpha=0.5)
    ax.axvline(gru_history["best_epoch"], color="r", linestyle="--", alpha=0.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Training & Validation Loss — LSTM vs GRU")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "loss_curves.png"), dpi=150)
    plt.close(fig)
    print(f"[evaluation] 损失曲线已保存至 {save_dir}/loss_curves.png")


def plot_predictions(y_true: np.ndarray,
                     lstm_pred: np.ndarray,
                     gru_pred: np.ndarray,
                     n_samples: int = 200,
                     save_dir: str = "figures"):
    """绘制预测值 vs 真实值对比（最近 N 个样本）"""
    os.makedirs(save_dir, exist_ok=True)

    # 取最后 n_samples 个样本
    if len(y_true) > n_samples:
        y_true = y_true[-n_samples:]
        lstm_pred = lstm_pred[-n_samples:]
        gru_pred = gru_pred[-n_samples:]

    fig, ax = plt.subplots(figsize=(14, 6))
    x_axis = np.arange(len(y_true))

    ax.plot(x_axis, y_true, "k-", linewidth=1.5, label="True Close")
    ax.plot(x_axis, lstm_pred, "b--", linewidth=1, alpha=0.8, label="LSTM Predict")
    ax.plot(x_axis, gru_pred, "r--", linewidth=1, alpha=0.8, label="GRU Predict")

    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Close Price (scaled)")
    ax.set_title("Prediction vs True Value — LSTM vs GRU")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "predictions.png"), dpi=150)
    plt.close(fig)
    print(f"[evaluation] 预测对比图已保存至 {save_dir}/predictions.png")


def plot_residuals(y_true: np.ndarray,
                   lstm_pred: np.ndarray,
                   gru_pred: np.ndarray,
                   save_dir: str = "figures"):
    """绘制残差分布直方图"""
    os.makedirs(save_dir, exist_ok=True)

    lstm_res = (y_true - lstm_pred)
    gru_res = (y_true - gru_pred)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(lstm_res, bins=40, color="b", alpha=0.7, edgecolor="white")
    axes[0].axvline(0, color="k", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Residual")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(f"LSTM Residual Distribution\nmean={lstm_res.mean():.4f}  std={lstm_res.std():.4f}")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(gru_res, bins=40, color="r", alpha=0.7, edgecolor="white")
    axes[1].axvline(0, color="k", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"GRU Residual Distribution\nmean={gru_res.mean():.4f}  std={gru_res.std():.4f}")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "residuals.png"), dpi=150)
    plt.close(fig)
    print(f"[evaluation] 残差分布图已保存至 {save_dir}/residuals.png")


# ==================== 主流程 ====================

def evaluate(data_dir: str = "data",
             model_dir: str = "models",
             save_dir: str = "figures"):
    """
    完整评估流程：
    1. 加载测试数据 + scaler_info
    2. 加载 LSTM / GRU 最优模型
    3. 计算预测值 + 反标准化
    4. 输出 5 项指标对比表
    5. 生成 3 张可视化图表
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 1. 加载数据 ----
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    info = np.load(os.path.join(data_dir, "scaler_info.npz"))
    close_mean, close_std = float(info["close_mean"]), float(info["close_std"])

    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    # ---- 2. 加载模型 ----
    # LSTM
    lstm = LSTMModel(input_size=X_test.shape[2])
    lstm.load_state_dict(torch.load(os.path.join(model_dir, "lstm_best.pt"),
                                    map_location=device, weights_only=True))
    lstm.to(device)
    lstm.eval()

    # GRU
    gru = GRUModel(input_size=X_test.shape[2])
    gru.load_state_dict(torch.load(os.path.join(model_dir, "gru_best.pt"),
                                   map_location=device, weights_only=True))
    gru.to(device)
    gru.eval()

    # ---- 3. 预测 + 反标准化 ----
    with torch.no_grad():
        lstm_pred_scaled = lstm(X_test_t.to(device)).cpu().numpy()
        gru_pred_scaled = gru(X_test_t.to(device)).cpu().numpy()

    y_true_real = inverse_scale(y_test, close_mean, close_std)
    lstm_pred_real = inverse_scale(lstm_pred_scaled, close_mean, close_std)
    gru_pred_real = inverse_scale(gru_pred_scaled, close_mean, close_std)

    # ---- 4. 指标计算 ----
    lstm_metrics = calc_metrics(y_true_real, lstm_pred_real)
    gru_metrics = calc_metrics(y_true_real, gru_pred_real)

    print("\n" + "=" * 60)
    print("                    评估结果对比")
    print("=" * 60)
    print(f"{'指标':<8} {'LSTM':>14} {'GRU':>14} {'较优':>10}")
    print("-" * 50)
    for key in ["MSE", "RMSE", "MAE", "MAPE", "R2"]:
        l_val = lstm_metrics[key]
        g_val = gru_metrics[key]
        if key == "R2":
            better = "LSTM *" if l_val > g_val else "GRU *"
        else:
            better = "LSTM *" if l_val < g_val else "GRU *"
        print(f"{key:<8} {l_val:>14.4f} {g_val:>14.4f} {better:>10}")
    print("=" * 60)

    # ---- 5. 可视化 ----
    # 加载训练历史
    lstm_history = dict(np.load(os.path.join(model_dir, "lstm_history.npz")))
    gru_history = dict(np.load(os.path.join(model_dir, "gru_history.npz")))

    plot_loss_curves(lstm_history, gru_history, save_dir)
    plot_predictions(y_true_real, lstm_pred_real, gru_pred_real, save_dir=save_dir)
    plot_residuals(y_true_real, lstm_pred_real, gru_pred_real, save_dir=save_dir)

    # ---- 6. 导出 JSON（供 Web Dashboard 使用） ----
    export_json(lstm_metrics, gru_metrics,
                lstm_history, gru_history,
                y_true_real, lstm_pred_real, gru_pred_real,
                save_dir=save_dir)

    return lstm_metrics, gru_metrics


def export_json(lstm_metrics: dict, gru_metrics: dict,
                lstm_history: dict, gru_history: dict,
                y_true: np.ndarray, lstm_pred: np.ndarray, gru_pred: np.ndarray,
                save_dir: str = "figures"):
    """
    导出 metrics.json + 图表数据 JSON，供 Web Dashboard 使用。
    """
    import json

    # ---- metrics.json ----
    metrics_data = {
        "LSTM": {k: round(float(v), 4) for k, v in lstm_metrics.items()},
        "GRU": {k: round(float(v), 4) for k, v in gru_metrics.items()},
        "LSTM_params": 34081,
        "GRU_params": 25569,
        "LSTM_best_epoch": int(lstm_history["best_epoch"]),
        "GRU_best_epoch": int(gru_history["best_epoch"]),
    }
    with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, ensure_ascii=False, indent=2)

    # ---- chart_history.json (损失曲线) ----
    history_data = {
        "lstm_train": [round(float(x), 6) for x in lstm_history["train_losses"]],
        "lstm_val": [round(float(x), 6) for x in lstm_history["val_losses"]],
        "gru_train": [round(float(x), 6) for x in gru_history["train_losses"]],
        "gru_val": [round(float(x), 6) for x in gru_history["val_losses"]],
        "lstm_best_epoch": int(lstm_history["best_epoch"]),
        "gru_best_epoch": int(gru_history["best_epoch"]),
    }
    with open(os.path.join(save_dir, "chart_history.json"), "w", encoding="utf-8") as f:
        json.dump(history_data, f, ensure_ascii=False, indent=2)

    # ---- chart_predict.json (预测对比，取最后 200 个点) ----
    n = min(200, len(y_true))
    predict_data = {
        "y_true": [round(float(x), 2) for x in y_true[-n:]],
        "lstm_pred": [round(float(x), 2) for x in lstm_pred[-n:]],
        "gru_pred": [round(float(x), 2) for x in gru_pred[-n:]],
    }
    with open(os.path.join(save_dir, "chart_predict.json"), "w", encoding="utf-8") as f:
        json.dump(predict_data, f, ensure_ascii=False, indent=2)

    # ---- chart_residual.json (残差分布) ----
    lstm_res = (y_true - lstm_pred).tolist()
    gru_res = (y_true - gru_pred).tolist()
    residual_data = {
        "lstm": [round(float(x), 4) for x in lstm_res],
        "gru": [round(float(x), 4) for x in gru_res],
        "lstm_mean": round(float(np.mean(lstm_res)), 4),
        "lstm_std": round(float(np.std(lstm_res)), 4),
        "gru_mean": round(float(np.mean(gru_res)), 4),
        "gru_std": round(float(np.std(gru_res)), 4),
    }
    with open(os.path.join(save_dir, "chart_residual.json"), "w", encoding="utf-8") as f:
        json.dump(residual_data, f, ensure_ascii=False, indent=2)

    print(f"[evaluation] JSON 数据已导出至 {save_dir}/ (4 个文件)")


if __name__ == "__main__":
    evaluate()


# ==================== 未来预测 ====================

def forecast_future(model, X_last: np.ndarray, close_mean: float, close_std: float,
                    steps: int = 15) -> list:
    """
    自回归滚动预测未来 N 天。

    工作原理：
    1. 用最后 30 天真实特征窗口预测第 T+1 天收盘价
    2. 将预测值填入 Close 列（索引 3），其余特征沿用上一天的值
    3. 窗口向前滑动 1 天，重复预测，共 steps 步

    Args:
        model: 训练好的 PyTorch 模型（LSTM 或 GRU）
        X_last: shape (30, 17) 最后 30 天标准化特征
        close_mean/close_std: 反标准化参数
        steps: 预测天数

    Returns:
        list[float]: 反标准化后的预测收盘价
    """
    device = next(model.parameters()).device
    model.eval()

    window = X_last.copy()  # (30, 17)
    predictions_scaled = []

    with torch.no_grad():
        for _ in range(steps):
            inp = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
            pred = model(inp).item()
            predictions_scaled.append(pred)

            # 构造下一时间步的特征窗口（滑动 + 用预测值替换 Close）
            new_row = window[-1].copy()
            new_row[3] = pred  # Close 在列索引 3
            window = np.vstack([window[1:], new_row])

    return [p * close_std + close_mean for p in predictions_scaled]


def export_forecast_json(data_dir: str = "data",
                         model_dir: str = "models",
                         save_dir: str = "figures"):
    """
    生成未来 15 天预测 JSON，供 Web 前端使用。

    流程：
    1. 加载原始数据 + 构建特征 → 取最后 30 天标准化特征
    2. 加载 LSTM / GRU 最优模型
    3. 自回归预测 15 天
    4. 导出 forecast.json
    """
    from feature_engineering import build_features, FEATURE_COLUMNS
    from model_lstm import LSTMModel
    from model_gru import GRUModel

    # ---- 1. 加载数据并构建特征 ----
    csv_path = os.path.join(data_dir, "600519_daily.csv")
    if not os.path.exists(csv_path):
        print(f"[forecast] 错误: 数据文件不存在 {csv_path}")
        return None

    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = build_features(df)
    df = df.dropna().reset_index(drop=True)

    feature_data = df[FEATURE_COLUMNS].values.astype(np.float64)

    # ---- 2. 加载 Scaler ----
    with open(os.path.join(model_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    # 取最后 30 天真实特征 + 标准化
    X_last = scaler.transform(feature_data[-30:])  # (30, 17)

    # 反标准化参数
    close_mean = float(scaler.mean_[3])
    close_std = float(scaler.scale_[3])

    # 历史收盘价（最近 60 天真实价格，供图表展示）
    history_close = [round(float(x), 2) for x in df["Close"].values[-60:]]
    history_dates = df["Date"].dt.strftime("%Y-%m-%d").values[-60:].tolist()

    # ---- 3. 加载模型 ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_size = len(FEATURE_COLUMNS)  # 18

    lstm = LSTMModel(input_size=input_size)
    lstm_path = os.path.join(model_dir, "lstm_best.pt")
    if not os.path.exists(lstm_path):
        print(f"[forecast] 错误: 模型文件不存在 {lstm_path}")
        return None
    lstm.load_state_dict(torch.load(lstm_path, map_location=device))
    lstm.to(device)
    lstm.eval()

    gru = GRUModel(input_size=input_size)
    gru_path = os.path.join(model_dir, "gru_best.pt")
    if not os.path.exists(gru_path):
        print(f"[forecast] 错误: 模型文件不存在 {gru_path}")
        return None
    gru.load_state_dict(torch.load(gru_path, map_location=device))
    gru.to(device)
    gru.eval()

    # ---- 4. 自回归预测 ----
    lstm_pred = forecast_future(lstm, X_last, close_mean, close_std, steps=15)
    gru_pred = forecast_future(gru, X_last, close_mean, close_std, steps=15)

    # ---- 5. 生成未来日期（跳过周末） ----
    last_date = df["Date"].iloc[-1]
    future_dates = []
    current = last_date
    while len(future_dates) < 15:
        current = current + pd.Timedelta(days=1)
        if current.weekday() < 5:  # 周一至周五
            future_dates.append(current.strftime("%Y-%m-%d"))

    # ---- 6. 导出 JSON ----
    forecast = {
        "lstm": [round(float(x), 2) for x in lstm_pred],
        "gru": [round(float(x), 2) for x in gru_pred],
        "dates": future_dates,
        "history_close": history_close,
        "history_dates": history_dates,
        "last_close": round(float(df["Close"].iloc[-1]), 2),
        "last_date": df["Date"].iloc[-1].strftime("%Y-%m-%d"),
    }

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "forecast.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(forecast, f, ensure_ascii=False, indent=2)

    print(f"[forecast] 未来 15 天预测已导出至 {out_path}")
    print(f"[forecast] LSTM 预测范围: {lstm_pred[0]:.2f} ~ {lstm_pred[-1]:.2f}")
    print(f"[forecast] GRU  预测范围: {gru_pred[0]:.2f} ~ {gru_pred[-1]:.2f}")

    return forecast
