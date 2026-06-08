"""
model_lstm.py — LSTM 模型定义与训练
双层 LSTM + Dropout + Dense，EarlyStopping 保留最优权重
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ==================== 模型定义 ====================

class LSTMModel(nn.Module):
    """双层 LSTM 预测模型"""

    def __init__(self, input_size: int = 17, hidden1: int = 64, hidden2: int = 32,
                 dropout: float = 0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden2, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)                      # (batch, 30, 64)
        x, _ = self.lstm2(x)                      # (batch, 30, 32)
        x = x[:, -1, :]                            # 取最后时间步
        x = self.dropout(x)
        x = self.fc(x)                             # (batch, 1)
        return x.squeeze(-1)


# ==================== 训练函数 ====================

def train_model(model: nn.Module,
                X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                epochs: int = 100,
                batch_size: int = 32,
                lr: float = 0.001,
                patience: int = 10,
                save_path: str = "models/lstm_best.pt",
                device: torch.device = None):
    """
    训练模型，使用 Early Stopping 保留验证集最优权重。

    Returns:
        dict: 包含 train_losses, val_losses, best_epoch
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    # 转换为 Tensor
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                              batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    print(f"[LSTM] 开始训练 | 设备:{device} | 样本:{X_train.shape[0]} | epochs:{epochs}")
    print(f"[LSTM] 参数量: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, epochs + 1):
        # ---- 训练阶段 ----
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)
        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # ---- 验证阶段 ----
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t.to(device))
            val_loss = criterion(val_preds, y_val_t.to(device)).item()
        val_losses.append(val_loss)

        # ---- 日志 ----
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | train_loss: {epoch_loss:.6f} | val_loss: {val_loss:.6f}")

        # ---- Early Stopping ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[LSTM] Early Stopping 触发于 epoch {epoch}，最优 epoch={best_epoch} val_loss={best_val_loss:.6f}")
                break

    print(f"[LSTM] 训练完成 | best_epoch={best_epoch} | best_val_loss={best_val_loss:.6f}")
    return {"train_losses": train_losses, "val_losses": val_losses, "best_epoch": best_epoch}


# ==================== 主入口 ====================

def run(data_dir: str = "data", save_path: str = "models/lstm_best.pt") -> dict:
    """完整 LSTM 训练流程：加载数据 → 训练 → 保存模型"""
    # 加载预处理数据
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val.npy"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(input_size=X_train.shape[2])

    history = train_model(model, X_train, y_train, X_val, y_val,
                          save_path=save_path, device=device)

    # 保存训练历史
    np.savez("models/lstm_history.npz",
             train_losses=history["train_losses"],
             val_losses=history["val_losses"],
             best_epoch=history["best_epoch"])

    return history


if __name__ == "__main__":
    run()
