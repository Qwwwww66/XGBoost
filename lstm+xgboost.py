"""
XGBoost + LSTM 联合压差预测与清洗决策（时间顺序 70/20/10 切分）

流程：
1) XGBoost 预测压差主趋势
2) 残差 = y_true - y_xgb_pred
3) LSTM 预测残差
4) y_final_pred = y_xgb_pred + y_lstm_residual_pred
5) 清洗决策：连续 WINDOW 个点 > THRESHOLD 判定 need_clean
"""

import math
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, Dataset

# 字体与设备
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "STSong", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 路径与列名
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "生产水处理系统压差变化数据.xlsx"
OUT_DIR = BASE_DIR / "output" / "lstm_xgb_pipeline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMNS = [
    "自清洗过滤器压差/kPa",
    "诱导阻截器A压差/kPa",
    "诱导阻截器B压差/kPa",
]

# 数据切分比例
TRAIN_RATIO = 0.7  # 可改为 0.8 搭配 VAL_RATIO=0.1 形成 8:1:1
VAL_RATIO = 0.2   # 剩余为测试

# XGBoost 配置
WINDOW_XGB = 12
XGB_PARAMS = dict(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=50,
    objective="reg:squarederror",
    random_state=42,
)

# LSTM 配置（仅用于“残差”序列：输入历史残差，输出下一时刻残差）
K_RES = 5  # 更短的残差窗口，提升样本可用性
EPOCHS = 20
BATCH_SIZE = 64
LR = 1e-3
HIDDEN_SIZE = 64
NUM_LAYERS = 1

# 清洗决策
THRESHOLD = 80.0
WINDOW_CLEAN = 5
MIN_TEST_RES = 10  # 测试集残差窗口至少保留的样本数，低于则跳过 LSTM 残差预测


# ========== 数据处理 ========== #
def load_data(file_path: Path) -> pd.DataFrame:
    df = pd.read_excel(file_path)
    df["TIME"] = pd.to_datetime(df["TIME"])
    df = df.sort_values("TIME").reset_index(drop=True)
    for col in TARGET_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def clean_series(df: pd.DataFrame, col: str) -> pd.Series:
    """
    基础清洗：去 NaN、去停机（<=0）、去明显清洗瞬态（压差快速下降）
    清洗瞬态：diff <= -30 kPa 视为清洗触发，删除该点及前后各 1 个点
    """
    s = pd.to_numeric(df[col], errors="coerce")
    s = s.fillna(method="ffill").fillna(method="bfill")
    mask = s > 0  # 去停机
    s = s.where(mask)

    diff = s.diff()
    cleaning = diff <= -30
    drop_idx = cleaning | cleaning.shift(1) | cleaning.shift(-1)
    s = s.mask(drop_idx)
    return s


def _norm_col(name: str) -> str:
    """用于宽松匹配列名：去掉空格、斜杠、下划线并转小写"""
    for ch in ["/", "\\", "_", " ", "\u3000"]:
        name = name.replace(ch, "")
    return name.lower()


def resolve_column(df: pd.DataFrame, target_col: str) -> str | None:
    if target_col in df.columns:
        return target_col
    norm_target = _norm_col(target_col)
    cand = None
    for c in df.columns:
        if _norm_col(c) == norm_target:
            cand = c
            break
    if cand:
        return cand
    for c in df.columns:
        if norm_target in _norm_col(c):
            cand = c
            break
    return cand


def create_lag_features(series: pd.Series, window: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    vals = series.values.astype(np.float32)
    times = series.index
    X, y, t = [], [], []
    for i in range(window, len(vals)):
        window_vals = vals[i - window : i]
        target_val = vals[i]
        if not np.isfinite(window_vals).all() or not np.isfinite(target_val):
            continue
        X.append(window_vals)
        y.append(target_val)
        t.append(times[i])
    if not X:
        return np.empty((0, window)), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=object)
    return (
        np.stack(X).reshape(-1, window).astype(np.float32),
        np.array(y, dtype=np.float32),
        np.array(t),
    )


def split_time_order(X, y, t, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    n = len(X)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    X_train, y_train, t_train = X[:n_train], y[:n_train], t[:n_train]
    X_val, y_val, t_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val], t[n_train : n_train + n_val]
    X_test, y_test, t_test = X[n_train + n_val :], y[n_train + n_val :], t[n_train + n_val :]
    print(f"[数据切分] train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    return X_train, y_train, t_train, X_val, y_val, t_val, X_test, y_test, t_test


# ========== LSTM 相关 ========== #
class ResidualDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMRegressor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.head(out)
        return out.squeeze(-1)


def build_residual_windows(residual: np.ndarray, times: np.ndarray, k: int):
    X, y, t = [], [], []
    for i in range(k, len(residual)):
        X.append(residual[i - k : i])
        y.append(residual[i])
        t.append(times[i])
    if not X:
        return np.empty((0, k, 1), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=object)
    X = np.stack(X).reshape(-1, k, 1).astype(np.float32)
    y = np.array(y, dtype=np.float32)
    t = np.array(t)
    return X, y, t


# ========== 训练函数 ========== #
def train_xgb(X_train, y_train, X_val, y_val):
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_lstm_residual(X_train, y_train, X_val, y_val):
    train_loader = DataLoader(ResidualDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(ResidualDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMRegressor(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        total /= len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            val_preds = []
            for xb, _ in val_loader:
                xb = xb.to(DEVICE)
                pred = model(xb).cpu().numpy()
                val_preds.append(pred)
            y_val_pred = np.concatenate(val_preds) if val_preds else np.array([])
            if len(y_val_pred) > 0:
                rmse = math.sqrt(mean_squared_error(y_val, y_val_pred))
                mae = mean_absolute_error(y_val, y_val_pred)
            else:
                rmse, mae = float("nan"), float("nan")
        if epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS:
            print(f"[LSTM] Epoch {epoch}/{EPOCHS}  Train MSE {total:.4f}  Val RMSE {rmse:.4f}  Val MAE {mae:.4f}")
    return model


# ========== 可视化 ========== #
def scatter_fit(y_true, y_pred, title: str, out_path: Path):
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
    plt.title(f"{title} (R={r:.3f}, R²={r2:.3f})", fontsize=13, pad=12)
    plt.xlabel("真实压差 y_true (kPa)", fontsize=12)
    plt.ylabel("预测压差 y_pred (kPa)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[可视化] 散点拟合图已保存: {out_path}")
    plt.show()


def plot_time_series(time_axis, y_true, y_pred, need_clean, threshold, title: str, out_path: Path):
    plt.figure(figsize=(12, 6))
    x_axis = np.arange(len(time_axis))
    plt.plot(x_axis, y_true, label="真实压差", color="#1f77b4", linewidth=2.0, alpha=0.85)
    plt.plot(x_axis, y_pred, label="最终预测压差", color="#ff7f0e", linewidth=2.0, alpha=0.85)
    triggers = np.where(need_clean)[0]
    if len(triggers) > 0:
        plt.scatter(triggers, y_pred[triggers], color="red", s=40, marker="o", label="清洗触发")
        for t in triggers:
            plt.axvline(x=t, color="red", linestyle=":", alpha=0.35, linewidth=1)
    plt.axhline(threshold, color="#2ca02c", linestyle="--", linewidth=1.5, label=f"阈值 {threshold} kPa")
    plt.title(title, fontsize=14, pad=12)
    plt.xlabel("时间序列索引（已按 TIME 排序）", fontsize=12)
    plt.ylabel("压差 (kPa)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.ticklabel_format(style="plain", axis="y")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[可视化] 清洗预警时间序列图已保存: {out_path}")
    plt.show()


# ========== 主流程 ========== #
def run_pipeline(target_col: str):
    print(f"\n{'='*60}\n处理列: {target_col}\n{'='*60}")
    df = load_data(DATA_FILE)
    resolved_col = resolve_column(df, target_col)
    if resolved_col is None:
        print(f"[错误] 未找到列: {target_col}，请检查数据列名")
        return
    if resolved_col != target_col:
        print(f"[提示] 目标列 '{target_col}' 映射为数据列 '{resolved_col}'")

    # 1) 原始数据清洗（去 NaN、停机、清洗瞬态）
    series = clean_series(df, resolved_col)
    series.index = df["TIME"]
    # 丢弃残留的 NaN/inf，避免标签含 NaN
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    # 基础数据概览
    nz = (series != 0).sum()
    print(f"[数据概览] 总样本 {len(series)}, 非零 {nz}, min={series.min():.4f}, max={series.max():.4f}, mean={series.mean():.4f}")

    # 2) 构造 XGBoost 滞后特征（在清洗后数据上，确保 float32）
    X_all, y_all, t_all = create_lag_features(series, window=WINDOW_XGB)
    # 3) 按时间顺序切分 train/val/test
    X_train, y_train, t_train, X_val, y_val, t_val, X_test, y_test, t_test = split_time_order(X_all, y_all, t_all)
    # 确保数值类型安全（XGBoost 不接受 object）
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_val = y_val.astype(np.float32)
    y_test = y_test.astype(np.float32)
    if len(X_test) == 0:
        print("[警告] 测试集为空，跳过")
        return

    xgb_model = train_xgb(X_train, y_train, X_val, y_val)
    y_train_pred = xgb_model.predict(X_train)
    y_val_pred = xgb_model.predict(X_val)
    y_test_pred = xgb_model.predict(X_test)

    rmse_xgb = math.sqrt(mean_squared_error(y_test, y_test_pred))
    mae_xgb = mean_absolute_error(y_test, y_test_pred)
    print(f"[XGB 测试集] RMSE: {rmse_xgb:.4f}  MAE: {mae_xgb:.4f}")

    safe_name = target_col.replace("/", "_")
    df_xgb_test = pd.DataFrame(
        {"TIME": pd.to_datetime(t_test), "y_true": y_test, "y_xgb_pred": y_test_pred}
    ).sort_values("TIME")
    xgb_csv = OUT_DIR / f"xgb_test_{safe_name}.csv"
    df_xgb_test.to_csv(xgb_csv, index=False, encoding="utf-8-sig")
    print(f"[输出] XGB 测试集预测 CSV: {xgb_csv}")

    # 4) 残差仅在切分后各自子集内处理；禁止切分前做窗口
    #    LSTM 目标：输入历史残差，预测下一时刻残差（不直接预测压差）
    resid_train = y_train - y_train_pred
    resid_val = y_val - y_val_pred
    resid_test = y_test - y_test_pred

    Xr_train, yr_train, tr_train = build_residual_windows(resid_train, t_train, K_RES)
    Xr_val, yr_val, tr_val = build_residual_windows(resid_val, t_val, K_RES)
    Xr_test, yr_test, tr_test = build_residual_windows(resid_test, t_test, K_RES)
    print(f"[残差窗口] train={len(Xr_train)}, val={len(Xr_val)}, test={len(Xr_test)} (窗口={K_RES})")

    use_lstm = len(Xr_train) > 0 and len(Xr_val) > 0 and len(Xr_test) >= MIN_TEST_RES
    if not use_lstm:
        print(f"[提示] 残差窗口样本不足（至少需测试集 >= {MIN_TEST_RES}），跳过 LSTM 残差预测，直接使用 XGB 结果")
        df_final = df_xgb_test.copy()
        df_final["y_final_pred"] = df_final["y_xgb_pred"]
        # 清洗决策
        above = df_final["y_final_pred"] > THRESHOLD
        df_final["need_clean"] = (
            above.rolling(WINDOW_CLEAN, min_periods=WINDOW_CLEAN)
            .apply(lambda s: 1 if s.all() else 0, raw=False)
            .fillna(0)
            .astype(bool)
        )
        final_csv = OUT_DIR / f"final_prediction_{safe_name}.csv"
        df_final[["TIME", "y_true", "y_final_pred", "need_clean"]].to_csv(final_csv, index=False, encoding="utf-8-sig")
        print(f"[输出] 最终预测 + 清洗决策 CSV（仅 XGB）: {final_csv}")
        # 评估与可视化
        y_true_final = df_final["y_true"].values
        y_pred_final = df_final["y_final_pred"].values
        r = float(np.corrcoef(y_true_final, y_pred_final)[0, 1]) if len(y_true_final) > 1 else float("nan")
        r2 = r * r if not math.isnan(r) else float("nan")
        rmse_final = math.sqrt(mean_squared_error(y_true_final, y_pred_final))
        mae_final = mean_absolute_error(y_true_final, y_pred_final)
        print(f"[最终预测 测试集] RMSE: {rmse_final:.4f}  MAE: {mae_final:.4f}  R: {r:.4f}  R²: {r2:.4f}")

        scatter_path = OUT_DIR / f"scatter_final_{safe_name}.png"
        scatter_fit(y_true_final, y_pred_final, f"{target_col} 最终预测拟合（仅XGB）", scatter_path)

        timeline_path = OUT_DIR / f"timeline_final_{safe_name}.png"
        plot_time_series(
            df_final["TIME"].values,
            df_final["y_true"].values,
            df_final["y_final_pred"].values,
            df_final["need_clean"].values,
            THRESHOLD,
            f"{target_col} 清洗预警（测试集，仅XGB）",
            timeline_path,
        )
        return

    lstm_model = train_lstm_residual(Xr_train, yr_train, Xr_val, yr_val)
    lstm_model.eval()
    with torch.no_grad():
        y_res_pred = lstm_model(torch.tensor(Xr_test, dtype=torch.float32).to(DEVICE)).cpu().numpy()

    rmse_res = math.sqrt(mean_squared_error(yr_test, y_res_pred))
    mae_res = mean_absolute_error(yr_test, y_res_pred)
    print(f"[LSTM 残差测试集] RMSE: {rmse_res:.4f}  MAE: {mae_res:.4f}")

    df_res_test = pd.DataFrame(
        {"TIME": pd.to_datetime(tr_test), "residual_true": yr_test, "y_lstm_residual_pred": y_res_pred}
    ).sort_values("TIME")
    res_csv = OUT_DIR / f"lstm_residual_test_{safe_name}.csv"
    df_res_test.to_csv(res_csv, index=False, encoding="utf-8-sig")
    print(f"[输出] 残差测试集预测 CSV: {res_csv}")

    # 融合
    df_final = pd.merge(df_xgb_test, df_res_test, on="TIME", how="inner")
    df_final["y_final_pred"] = df_final["y_xgb_pred"] + df_final["y_lstm_residual_pred"]

    y_true_final = df_final["y_true"].values
    y_pred_final = df_final["y_final_pred"].values
    n_test_final = len(y_true_final)
    if n_test_final >= 10:
        r = float(np.corrcoef(y_true_final, y_pred_final)[0, 1])
        r2 = r * r if not math.isnan(r) else float("nan")
    else:
        r, r2 = float("nan"), float("nan")
        print(f"[警告] 测试样本数 {n_test_final} < 10，跳过相关系数 R/R² 计算与散点图")
    rmse_final = math.sqrt(mean_squared_error(y_true_final, y_pred_final))
    mae_final = mean_absolute_error(y_true_final, y_pred_final)
    print(f"[最终预测 测试集] RMSE: {rmse_final:.4f}  MAE: {mae_final:.4f}  R: {r:.4f}  R²: {r2:.4f}")

    # 清洗决策（仅基于测试集预测 y_final_pred，不使用真实值触发）
    above = df_final["y_final_pred"] > THRESHOLD
    df_final["need_clean"] = (
        above.rolling(WINDOW_CLEAN, min_periods=WINDOW_CLEAN)
        .apply(lambda s: 1 if s.all() else 0, raw=False)
        .fillna(0)
        .astype(bool)
    )

    final_csv = OUT_DIR / f"final_prediction_{safe_name}.csv"
    df_final[["TIME", "y_true", "y_final_pred", "need_clean"]].to_csv(final_csv, index=False, encoding="utf-8-sig")
    print(f"[输出] 最终预测 + 清洗决策 CSV: {final_csv}")

    # 可视化
    if n_test_final >= 10:
        scatter_path = OUT_DIR / f"scatter_final_{safe_name}.png"
        scatter_fit(y_true_final, y_pred_final, f"{target_col} 最终预测拟合", scatter_path)

    timeline_path = OUT_DIR / f"timeline_final_{safe_name}.png"
    plot_time_series(
        df_final["TIME"].values,
        df_final["y_true"].values,
        df_final["y_final_pred"].values,
        df_final["need_clean"].values,
        THRESHOLD,
        f"{target_col} 清洗预警（测试集）",
        timeline_path,
    )


def main():
    for col in TARGET_COLUMNS:
        run_pipeline(col)


if __name__ == "__main__":
    main()
