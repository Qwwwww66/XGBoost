"""
XGBoost regression for pressure drop with validation/test correlation plots (SCI style).
Outputs:
- Validation and test scatter plots with regression line and y=x reference.
- Scatter data Excel per target (validation + test).
"""

import math
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# 路径设置
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "生产水处理系统压差变化数据.xlsx"
OUT_PLOT_DIR = BASE_DIR / "xgboost_results"
OUT_CSV_DIR = BASE_DIR / "output" / "xgb_predictions"
OUT_PLOT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV_DIR.mkdir(parents=True, exist_ok=True)

# 目标列与英文名称映射
TARGET_COLUMNS = [
    "诱导阻截器A压差/kPa",
    "诱导阻截器B压差/kPa",
    "自清洗过滤器压差/kPa",
]
EN_NAMES = {
    "诱导阻截器A压差/kPa": "Inducer A",
    "诱导阻截器B压差/kPa": "Inducer B",
    "自清洗过滤器压差/kPa": "Self-cleaning filter",
}

WINDOW_SIZE = 12  # 滑窗长度


def load_data(file_path: Path) -> pd.DataFrame:
    """读取与基础预处理：按时间排序，数值列前向/后向填充。"""
    df = pd.read_excel(file_path)
    df["TIME"] = pd.to_datetime(df["TIME"])
    df = df.sort_values("TIME").reset_index(drop=True)
    for col in TARGET_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(method="ffill").fillna(method="bfill")
    return df


def extract_dynamic_segments(df: pd.DataFrame, column_name: str, min_length=15, max_p=280, plateau_std=0.03):
    """提取上升段：过滤停机、去平台，保留 TIME 和目标列。"""
    mask = df[column_name] > 0.5
    group_ids = (mask != mask.shift()).cumsum()
    valid_groups = group_ids[mask]
    segments: List[pd.DataFrame] = []
    for g_id in valid_groups.unique():
        seg = df.loc[group_ids == g_id, ["TIME", column_name]].copy()
        rolling_diff = seg[column_name].diff().abs().rolling(window=5).mean().fillna(1)
        is_plateau = (seg[column_name] > 50) & (rolling_diff < plateau_std)
        if is_plateau.any():
            cut_point = is_plateau.idxmax()
            seg = seg.loc[:cut_point].iloc[:-1]
        seg = seg[seg[column_name] < max_p]
        if len(seg) >= min_length:
            segments.append(seg)
    return segments


def create_features(segments: List[pd.DataFrame], window_size=WINDOW_SIZE) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """滑窗特征：过去 window 个压差 + 时间步，预测当前压差；返回 X, y, time。"""
    X, y, times = [], [], []
    for seg in segments:
        vals = seg.iloc[:, 1].values
        tvals = seg["TIME"].values
        for i in range(window_size, len(vals)):
            lags = vals[i - window_size : i]
            time_step = i
            X.append(np.append(lags, time_step))
            y.append(vals[i])
            times.append(tvals[i])
    if not X:
        return np.empty((0, window_size + 1)), np.empty((0,)), np.empty((0,))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(times)


def split_time_order(X, y, t, train_ratio=0.7, val_ratio=0.2):
    """按时间顺序切分 train/val/test（默认 7/2/1），并转为 float32。"""
    n = len(X)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    X_train, y_train, t_train = X[:n_train], y[:n_train], t[:n_train]
    X_val, y_val, t_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val], t[n_train : n_train + n_val]
    X_test, y_test, t_test = X[n_train + n_val :], y[n_train + n_val :], t[n_train + n_val :]
    return (
        X_train.astype(np.float32),
        y_train.astype(np.float32),
        t_train,
        X_val.astype(np.float32),
        y_val.astype(np.float32),
        t_val,
        X_test.astype(np.float32),
        y_test.astype(np.float32),
        t_test,
    )


def plot_scatter(y_true, y_pred, dataset_name: str, target_name_en: str, out_path: Path):
    """相关性散点图（回归线、y=x、R/R²/RMSE/n，SCI 英文风格）。"""
    if len(y_true) < 2:
        return
    for style_name in ["seaborn-v0_8-whitegrid", "seaborn-whitegrid", "default"]:
        try:
            plt.style.use(style_name)
            break
        except Exception:
            continue
    r = float(np.corrcoef(y_true, y_pred)[0, 1])
    r2 = r * r if not math.isnan(r) else float("nan")
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    n_samples = len(y_true)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, color="#1f77b4", edgecolors="none", label=f"{dataset_name} samples")
    slope, intercept = np.polyfit(y_true, y_pred, 1)
    x_line = np.array([y_true.min(), y_true.max()])
    y_line = slope * x_line + intercept
    plt.plot(x_line, y_line, color="#1f77b4", linestyle="-", linewidth=2, label=f"Fit: y = {slope:.3f}x + {intercept:.3f}")
    plt.plot(x_line, x_line, color="gray", linestyle="--", linewidth=1, label="y = x")
    plt.title(
        f"{dataset_name} set: Predicted vs. Measured Pressure Drop of {target_name_en}\n"
        f"(R = {r:.3f}, R² = {r2:.3f}, RMSE = {rmse:.2f} kPa, n = {n_samples})",
        fontsize=13,
        pad=12,
    )
    plt.xlabel("Measured pressure drop, y_true (kPa)", fontsize=12)
    plt.ylabel("Predicted pressure drop, y_pred (kPa)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.text(
        0.05,
        0.95,
        f"y = {slope:.3f}x + {intercept:.3f}\nRMSE = {rmse:.2f} kPa\nn = {n_samples}",
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"),
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def safe_name(target: str) -> str:
    name = target.replace("/kPa", "").replace("/", "_").replace("\\", "_").replace(" ", "")
    return name


def run_pipeline(file_path: Path, target_col: str):
    df = load_data(file_path)
    segments = extract_dynamic_segments(df, target_col, max_p=280, plateau_std=0.03)
    if len(segments) == 0:
        return

    X, y, times = create_features(segments, window_size=WINDOW_SIZE)
    if len(X) < 10:
        return

    X_train, y_train, t_train, X_val, y_val, t_val, X_test, y_test, t_test = split_time_order(X, y, times)

    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=50,
        eval_metric="rmse",
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # 导出散点数据（验证+测试）到 Excel
    df_val = pd.DataFrame({"dataset": "validation", "y_true": y_val, "y_pred": y_val_pred})
    df_test = pd.DataFrame({"dataset": "test", "y_true": y_test, "y_pred": y_test_pred})
    scatter_excel_path = OUT_CSV_DIR / f"scatter_data_{safe_name(target_col)}.xlsx"
    pd.concat([df_val, df_test], ignore_index=True).to_excel(scatter_excel_path, index=False)

    obj_en = EN_NAMES.get(target_col, target_col)
    plot_scatter(y_val, y_val_pred, "Validation", obj_en, OUT_PLOT_DIR / f"scatter_val_{safe_name(target_col)}.png")
    plot_scatter(y_test, y_test_pred, "Test", obj_en, OUT_PLOT_DIR / f"scatter_test_{safe_name(target_col)}.png")


def main():
    for col in TARGET_COLUMNS:
        run_pipeline(DATA_FILE, col)


if __name__ == "__main__":
    main()
