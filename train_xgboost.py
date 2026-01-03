import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

plt.style.use("ggplot")

# ===== 配置 =====
BASE_DIR = Path(__file__).resolve().parent  # models 目录
DATA_DIR = BASE_DIR.parent / "output" / "cleaned"
ENCODING = "utf-8-sig"

# 一次跑三台设备
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


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding=ENCODING, engine="python")
    if "TIME" in df.columns:
        df = df.sort_values("TIME").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    return df


def train_val_split(df: pd.DataFrame, ratio: float = 0.7):
    n = len(df)
    split_idx = int(n * ratio)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def build_model() -> XGBRegressor:
    # 基线参数
    return XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1.0,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=4,
        verbosity=1,
    )


def run_one(name: str, path: Path):
    if not path.exists():
        print(f"[{name}] 文件不存在: {path}")
        return

    df = load_data(path)
    train_df, val_df = train_val_split(df, ratio=0.7)
    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_val, y_val = val_df[FEATURE_COLS], val_df[TARGET_COL]

    model = build_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)

    print(f"\n== {name} ==")
    print(f"样本数: 训练 {len(train_df)}, 验证 {len(val_df)}")
    print(f"验证集 RMSE: {rmse:.4f}")
    print(f"验证集 MAE : {mae:.4f}")
    print("训练集 y 范围:", float(y_train.min()), float(y_train.max()))
    print("验证集 y 范围:", float(y_val.min()), float(y_val.max()))
    print("预测值范围   :", float(y_pred.min()), float(y_pred.max()))

    booster = model.get_booster()
    fscore = booster.get_score(importance_type="gain")
    print("特征重要性 (gain):")
    for k, v in sorted(fscore.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {k}: {v:.4f}")

    # 绘制验证集真实 vs 预测 + 残差
    x_axis = np.arange(len(y_val))
    y_true = y_val.to_numpy()

    # 动态 y 轴范围（1%-99% 分位）避免显示成直线
    p1, p99 = np.percentile(np.hstack([y_true, y_pred]), [1, 99])
    padding = max((p99 - p1) * 0.1, 1.0)
    y_min = p1 - padding
    y_max = p99 + padding

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax1.plot(x_axis, y_true, label="真实压差 y", linewidth=1.6, color="#1f77b4")
    ax1.plot(x_axis, y_pred, label="预测压差 y_pred", linewidth=1.4, color="#ff7f0e")
    ax1.set_ylabel("压差 (kPa)")
    ax1.set_ylim(y_min, y_max)
    ax1.ticklabel_format(style="plain", axis="y")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"{name} 验证集压差预测对比  (RMSE={rmse:.2f}, MAE={mae:.2f})")

    residuals = y_pred - y_true
    ax2.plot(x_axis, residuals, label="残差 (预测-真实)", linewidth=1.2, color="#2ca02c")
    ax2.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax2.set_xlabel("验证集样本序号（按时间顺序）")
    ax2.set_ylabel("残差")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    plt.tight_layout()
    plt.show()


def main():
    for name, path in FEATURE_FILES.items():
        run_one(name, path)


if __name__ == "__main__":
    main()
