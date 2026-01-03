import math
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

plt.style.use("ggplot")

# ===== 配置 =====
BASE_DIR = Path(__file__).resolve().parent  # models 目录
DATA_DIR = BASE_DIR.parent / "output" / "cleaned"
ENCODING = "utf-8-sig"

# 三台设备的特征文件
FEATURE_FILES = {
    "self_clean_filter": DATA_DIR / "self_clean_filter_features.csv",
    "inducer_A": DATA_DIR / "inducer_A_features.csv",
    "inducer_B": DATA_DIR / "inducer_B_features.csv",
}

FEATURE_COLS = [
    "pressure_lag_1",
    "pressure_lag_2",
    "pressure_lag_3",
    "delta_p",
    "cycle_step",
    "cycle_length",
    "cycle_progress",
]
TARGET_COL = "y"

# 训练超参
SEQ_LEN = 16           # LSTM 序列长度
EPOCHS = 20
BATCH_SIZE = 64
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SeqDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32),
        )


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取最后时间步
        out = self.head(out)
        return out.squeeze(-1)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding=ENCODING, engine="python")
    if "TIME" in df.columns:
        df = df.sort_values("TIME").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    return df


def train_val_split(df: pd.DataFrame, ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    split_idx = int(n * ratio)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def build_sequences(df: pd.DataFrame, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    按 cycle_id 生成滑窗序列，不跨周期。
    """
    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    for _, g in df.groupby("cycle_id"):
        g = g.reset_index(drop=True)
        if len(g) <= seq_len:
            continue
        feat = g[FEATURE_COLS].to_numpy()
        target = g[TARGET_COL].to_numpy()
        for i in range(seq_len, len(g)):
            X_list.append(feat[i - seq_len : i])
            y_list.append(target[i])
    if not X_list:
        return np.empty((0, seq_len, len(FEATURE_COLS))), np.empty((0,))
    return np.stack(X_list), np.array(y_list)


def run_one(name: str, path: Path):
    if not path.exists():
        print(f"[{name}] 文件不存在: {path}")
        return

    df = load_data(path)
    train_df, val_df = train_val_split(df, ratio=0.7)

    # 标准化特征与目标（在训练集上拟合，再应用到验证集）
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    train_df_scaled = train_df.copy()
    val_df_scaled = val_df.copy()

    train_df_scaled[FEATURE_COLS] = scaler_x.fit_transform(train_df[FEATURE_COLS])
    val_df_scaled[FEATURE_COLS] = scaler_x.transform(val_df[FEATURE_COLS])

    train_df_scaled[TARGET_COL] = scaler_y.fit_transform(train_df[[TARGET_COL]])
    val_df_scaled[TARGET_COL] = scaler_y.transform(val_df[[TARGET_COL]])

    X_train, y_train = build_sequences(train_df_scaled, SEQ_LEN)
    X_val, y_val = build_sequences(val_df_scaled, SEQ_LEN)

    if len(X_train) == 0 or len(X_val) == 0:
        print(f"[{name}] 序列样本不足，无法训练（检查 seq_len 或数据量）。")
        return

    train_loader = DataLoader(SeqDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(SeqDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMRegressor(input_size=len(FEATURE_COLS)).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                loss = criterion(model(xb), yb)
                val_loss += loss.item() * len(xb)
            val_loss /= len(val_loader.dataset)
        if epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS:
            print(f"[{name}] Epoch {epoch}/{EPOCHS} Train MSE {epoch_loss:.4f}  Val MSE {val_loss:.4f}")

    # 验证集预测
    model.eval()
    preds: List[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in val_loader:
            xb = xb.to(DEVICE)
            pred = model(xb).cpu().numpy()
            preds.append(pred)
    y_pred = np.concatenate(preds)

    # 反标准化回原始刻度
    y_true_orig = scaler_y.inverse_transform(y_val.reshape(-1, 1)).squeeze()
    y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).squeeze()

    rmse = math.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    mae = mean_absolute_error(y_true_orig, y_pred_orig)

    print(f"\n== {name} ==")
    print(f"样本数: 训练序列 {len(X_train)}, 验证序列 {len(X_val)}")
    print(f"验证集 RMSE: {rmse:.4f}")
    print(f"验证集 MAE : {mae:.4f}")
    print("验证集 y 范围:", float(y_true_orig.min()), float(y_true_orig.max()))
    print("预测值范围   :", float(y_pred_orig.min()), float(y_pred_orig.max()))

    # 绘制验证集真实 vs 预测 + 残差
    x_axis = np.arange(len(y_true_orig))
    y_true = y_true_orig
    y_pred_plot = y_pred_orig

    p1, p99 = np.percentile(np.hstack([y_true, y_pred_plot]), [1, 99])
    padding = max((p99 - p1) * 0.1, 1.0)
    y_min = p1 - padding
    y_max = p99 + padding

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax1.plot(x_axis, y_true, label="真实压差 y", linewidth=1.6, color="#1f77b4")
    ax1.plot(x_axis, y_pred_plot, label="预测压差 y_pred", linewidth=1.4, color="#ff7f0e")
    ax1.set_ylabel("压差 (kPa)")
    ax1.set_ylim(y_min, y_max)
    ax1.ticklabel_format(style="plain", axis="y")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"{name} LSTM 验证集压差预测  (RMSE={rmse:.2f}, MAE={mae:.2f})")

    residuals = y_pred_plot - y_true
    ax2.plot(x_axis, residuals, label="残差 (预测-真实)", linewidth=1.2, color="#2ca02c")
    ax2.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax2.set_xlabel("验证集样本序号（滑窗后序列）")
    ax2.set_ylabel("残差")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    plt.tight_layout()
    plt.show()


def main():
    print(f"使用设备: {DEVICE}")
    for name, path in FEATURE_FILES.items():
        run_one(name, path)


if __name__ == "__main__":
    main()
