import math
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, Dataset

# 字体设置，防止中文显示为方框
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "STSong", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# ===== 基本配置 =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 20
BATCH_SIZE = 64
LR = 1e-3
HIDDEN_SIZE = 64
NUM_LAYERS = 1
K_WINDOW = 16  # LSTM 滑动窗口长度
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2  # 剩余默认 0.1 为测试集

# 路径配置
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "生产水处理系统压差变化数据.xlsx"
OUT_PLOT_DIR = BASE_DIR / "output" / "lstm_pressure_results"
OUT_CSV_DIR = BASE_DIR / "output" / "lstm_pressure_predictions"
OUT_PLOT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMNS = [
    "自清洗过滤器压差/kPa",
    "诱导阻截器A压差/kPa",
    "诱导阻截器B压差/kPa",
]


# ===== 数据准备 =====
def load_data(file_path: Path) -> pd.DataFrame:
    print(f"正在读取文件: {file_path} ...")
    df = pd.read_excel(file_path)
    df["TIME"] = pd.to_datetime(df["TIME"])
    df = df.sort_values("TIME").reset_index(drop=True)
    for col in TARGET_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def extract_dynamic_segments(df: pd.DataFrame, column_name: str, min_length=15, max_p=280, plateau_std=0.03):
    """
    提取上升段，同时保留 TIME，避免长平台段。
    """
    mask = df[column_name] > 0.5
    group_ids = (mask != mask.shift()).cumsum()
    valid_groups = group_ids[mask]
    dynamic_segments = []
    for g_id in valid_groups.unique():
        seg = df.loc[group_ids == g_id, ["TIME", column_name]].copy()
        rolling_diff = seg[column_name].diff().abs().rolling(window=5).mean().fillna(1)
        is_plateau = (seg[column_name] > 50) & (rolling_diff < plateau_std)
        if is_plateau.any():
            cut_point = is_plateau.idxmax()
            seg = seg.loc[:cut_point].iloc[:-1]
        seg = seg[seg[column_name] < max_p]
        if len(seg) >= min_length:
            dynamic_segments.append(seg)
    print(f"[{column_name}] 提取到 {len(dynamic_segments)} 段有效上升序列")
    return dynamic_segments


def create_windows(segments: List[pd.DataFrame], k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成窗口：输入过去 k 个压差，输出下一个压差；同时保留 TIME。
    """
    X, y, times = [], [], []
    for seg in segments:
        vals = seg.iloc[:, 1].values.astype(np.float32)
        tvals = seg["TIME"].values
        for i in range(k, len(vals)):
            X.append(vals[i - k : i])
            y.append(vals[i])
            times.append(tvals[i])
    if not X:
        return np.empty((0, k, 1), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=object)
    X = np.stack(X).reshape(-1, k, 1)
    y = np.array(y, dtype=np.float32)
    times = np.array(times)
    return X, y, times


def split_train_val_test(X, y, t, train_ratio: float = TRAIN_RATIO, val_ratio: float = VAL_RATIO):
    n = len(X)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    X_train, y_train, t_train = X[:n_train], y[:n_train], t[:n_train]
    X_val, y_val, t_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val], t[n_train : n_train + n_val]
    X_test, y_test, t_test = X[n_train + n_val :], y[n_train + n_val :], t[n_train + n_val :]
    print(f"[数据切分] train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test, t_train, t_val, t_test


# ===== Dataset 与模型 =====
class PressureDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 1):
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
        out = out[:, -1, :]
        out = self.head(out)
        return out.squeeze(-1)


def train_lstm_pressure(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    hidden_size: int = HIDDEN_SIZE,
    num_layers: int = NUM_LAYERS,
):
    train_loader = DataLoader(PressureDataset(X_train, y_train), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(PressureDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    model = LSTMRegressor(input_size=1, hidden_size=hidden_size, num_layers=num_layers).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        total_loss /= len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            val_preds = []
            for xb, _ in val_loader:
                xb = xb.to(DEVICE)
                pred = model(xb).cpu().numpy()
                val_preds.append(pred)
            y_val_pred = np.concatenate(val_preds)
            rmse = math.sqrt(mean_squared_error(y_val, y_val_pred))
            mae = mean_absolute_error(y_val, y_val_pred)
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch}/{epochs}  Train MSE {total_loss:.4f}  Val RMSE {rmse:.4f}  Val MAE {mae:.4f}")

    return model, y_val_pred, rmse, mae


def plot_prediction(name: str, y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path, rmse: float, mae: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    x_axis = np.arange(len(y_true))
    plt.figure(figsize=(12, 5))
    plt.plot(x_axis, y_true, label="真实压差", linewidth=2.0, color="#1f77b4", alpha=0.85)
    plt.plot(x_axis, y_pred, label="LSTM预测压差", linewidth=2.0, color="#d62728", linestyle="--")
    plt.title(f"{name} LSTM 压差预测 (RMSE={rmse:.2f}, MAE={mae:.2f})", fontsize=16, pad=14)
    plt.xlabel("序列索引（时间顺序）", fontsize=14)
    plt.ylabel("压差 (kPa)", fontsize=14)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.ticklabel_format(style="plain", axis="y")
    out_path = out_dir / f"lstm_pressure_{name.replace('/', '_')}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[可视化] 已保存: {out_path}")
    plt.show()


def plot_scatter_fit(name: str, y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(y_true) > 1:
        r = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        r = float("nan")
    r2 = r * r if not math.isnan(r) else float("nan")

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, color="#1f77b4", edgecolors="white", linewidths=0.6, label="测试集样本")
    min_v = min(y_true.min(), y_pred.min())
    max_v = max(y_true.max(), y_pred.max())
    plt.plot([min_v, max_v], [min_v, max_v], color="#d62728", linestyle="--", linewidth=2, label="y = x 参考线")
    plt.title(f"{name} 测试集拟合 (R={r:.3f}, R²={r2:.3f})", fontsize=13, pad=12)
    plt.xlabel("真实压差 y_true (kPa)", fontsize=12)
    plt.ylabel("预测压差 y_pred (kPa)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    out_path = out_dir / f"lstm_pressure_scatter_{name.replace('/', '_')}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[可视化] 测试集散点拟合图已保存: {out_path}")
    plt.show()


def save_prediction_csv(name: str, times: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray):
    df_out = pd.DataFrame(
        {
            "TIME": pd.to_datetime(times),
            "y_true": y_true,
            "y_lstm_pred": y_pred,
        }
    ).sort_values("TIME")
    safe_name = name.replace("/", "_")

    # 清洗决策
    THRESHOLD = 80.0
    WINDOW = 5
    above = df_out["y_lstm_pred"] > THRESHOLD
    df_out["need_clean"] = (
        above.rolling(WINDOW, min_periods=WINDOW)
        .apply(lambda s: 1 if s.all() else 0, raw=False)
        .fillna(0)
        .astype(bool)
    )

    out_path = OUT_CSV_DIR / f"lstm_pressure_pred_{safe_name}.csv"
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[输出] LSTM 压差预测 CSV 已保存: {out_path}")

    # 额外输出测试集命名示例：lstm_test_predictions_<列名>.csv
    test_path = OUT_CSV_DIR / f"lstm_test_predictions_{safe_name}.csv"
    df_out[["TIME", "y_lstm_pred", "need_clean"]].to_csv(test_path, index=False, encoding="utf-8-sig")
    print(f"[输出] LSTM 测试集预测 CSV 已保存: {test_path} (含 need_clean)")


# ===== 主流程：直接用 LSTM 预测压差 =====
if __name__ == "__main__":
    for col in TARGET_COLUMNS:
        print(f"\n{'='*60}\n处理列: {col}\n{'='*60}")
        df = load_data(DATA_FILE)
        segments = extract_dynamic_segments(df, col, max_p=280, plateau_std=0.03)
        if len(segments) == 0:
            print(f"[警告] 未提取到有效序列，跳过 {col}")
            continue

        X, y, times = create_windows(segments, k=K_WINDOW)
        if len(X) == 0:
            print(f"[警告] 窗口生成为空，跳过 {col}")
            continue

        X_train, X_val, X_test, y_train, y_val, y_test, t_train, t_val, t_test = split_train_val_test(
            X, y, times, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO
        )

        model, y_val_pred, rmse_val, mae_val = train_lstm_pressure(
            X_train, y_train, X_val, y_val, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS
        )
        # 测试集评估
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
            y_test_pred = model(X_test_tensor).cpu().numpy()
        rmse_test = math.sqrt(mean_squared_error(y_test, y_test_pred))
        mae_test = mean_absolute_error(y_test, y_test_pred)
        print(f"[验证集] RMSE: {rmse_val:.4f}  MAE: {mae_val:.4f}")
        print(f"[测试集]  RMSE: {rmse_test:.4f}  MAE: {mae_test:.4f}")

        # 可视化与输出使用测试集
        plot_prediction(col, y_test, y_test_pred, OUT_PLOT_DIR, rmse_test, mae_test)
        plot_scatter_fit(col, y_test, y_test_pred, OUT_PLOT_DIR)
        save_prediction_csv(col, t_test, y_test, y_test_pred)
