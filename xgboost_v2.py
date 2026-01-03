import os
import math
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 字体设置，防止中文显示为方框
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "STSong", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# 路径与配置
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "生产水处理系统压差变化数据.xlsx"
OUT_PLOT_DIR = BASE_DIR / "xgboost_results"
OUT_CSV_DIR = BASE_DIR / "output" / "xgb_predictions"
OUT_PLOT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMNS = [
    "自清洗过滤器压差/kPa",
    "诱导阻截器A压差/kPa",
    "诱导阻截器B压差/kPa",
]

WINDOW_SIZE = 12  # 滑窗长度


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
    提取上升段，并保留 TIME。
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


def create_features(segments: List[pd.DataFrame], window_size=WINDOW_SIZE) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    返回 X, y, times，对应 y 的时间戳
    """
    X, y, times = [], [], []
    for seg in segments:
        vals = seg.iloc[:, 1].values  # 压差
        tvals = seg["TIME"].values
        for i in range(window_size, len(vals)):
            lags = vals[i - window_size : i]
            time_step = i
            X.append(np.append(lags, time_step))
            y.append(vals[i])
            times.append(tvals[i])
    return np.array(X), np.array(y), np.array(times)


def run_model_pipeline(file_path: Path, target_col: str):
    df = load_data(file_path)
    segments = extract_dynamic_segments(df, target_col, max_p=280, plateau_std=0.03)
    if len(segments) == 0:
        print(f"[{target_col}] 未提取到有效数据片段，跳过")
        return None

    X, y, times = create_features(segments, window_size=WINDOW_SIZE)

    # time-ordered split: 70% train, 20% val, 10% test
    n = len(X)
    n_train = int(n * 0.7)
    n_val = int(n * 0.2)
    X_train, y_train, times_train = X[:n_train], y[:n_train], times[:n_train]
    X_val, y_val, times_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val], times[n_train : n_train + n_val]
    X_test, y_test, times_test = X[n_train + n_val :], y[n_train + n_val :], times[n_train + n_val :]
    print(
        f"[数据切分] train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)} "
        f"(比例约 {len(X_train)/n:.2f}/{len(X_val)/n:.2f}/{len(X_test)/n:.2f})"
    )

    print("\n[模型训练] 正在训练...")
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=50,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_pred = model.predict(X_test)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    # 测试集相关系数
    if len(y_test) > 1:
        r = float(np.corrcoef(y_test, y_pred)[0, 1])
    else:
        r = float("nan")
    r2 = r * r if not math.isnan(r) else float("nan")

    print(f"\n[评估结果] {target_col}")
    print(f"测试集 RMSE: {rmse:.4f} kPa")
    print(f"测试集 MAE : {mae:.4f} kPa")
    print(f"测试集 Pearson R: {r:.4f}, R^2: {r2:.4f}")

    # 保存预测 CSV：TIME, y_xgb_pred（原有文件）
    df_out = pd.DataFrame(
        {
            "TIME": pd.to_datetime(times_test),
            "y_xgb_pred": y_pred,
        }
    ).sort_values("TIME")
    safe_col_name = target_col.replace("/", "_").replace("\\", "_")
    csv_path = OUT_CSV_DIR / f"xgb_pred_{safe_col_name}.csv"
    df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[输出] 预测 CSV 已保存: {csv_path}")

    # 额外输出测试集完整结果：TIME, y_true, y_pred
    df_test = pd.DataFrame(
        {
            "TIME": pd.to_datetime(times_test),
            "y_true": y_test,
            "y_pred": y_pred,
        }
    ).sort_values("TIME")
    test_csv_path = OUT_CSV_DIR / f"xgboost_test_predictions_{safe_col_name}.csv"
    df_test.to_csv(test_csv_path, index=False, encoding="utf-8-sig")
    print(f"[输出] 测试集预测 CSV 已保存: {test_csv_path}")

    # 简易清洗决策：连续 WINDOW 个点都大于 THRESHOLD 判定为 need_clean
    THRESHOLD = 80.0
    WINDOW = 5
    above = df_test["y_pred"] > THRESHOLD
    df_test["need_clean"] = (
        above.rolling(WINDOW, min_periods=WINDOW)
        .apply(lambda s: 1 if s.all() else 0, raw=False)
        .fillna(0)
        .astype(bool)
    )
    decision_csv_path = OUT_CSV_DIR / f"xgboost_test_predictions_clean_{safe_col_name}.csv"
    df_test[["TIME", "y_pred", "need_clean"]].to_csv(decision_csv_path, index=False, encoding="utf-8-sig")
    print(f"[输出] 清洗决策 CSV 已保存: {decision_csv_path} (阈值={THRESHOLD}, 连续窗口={WINDOW})")

    # 清洗预警场景可视化（时间序列）
    plt.figure(figsize=(12, 6))
    x_axis = np.arange(len(df_test))
    plt.plot(x_axis, df_test["y_true"], label="真实压差", color="#1f77b4", linewidth=2.0, alpha=0.85)
    plt.plot(x_axis, df_test["y_pred"], label="XGBoost预测压差", color="#ff7f0e", linewidth=2.0, alpha=0.85)
    triggers = df_test.index[df_test["need_clean"]]
    if len(triggers) > 0:
        plt.scatter(triggers, df_test.loc[triggers, "y_pred"], color="red", s=40, marker="o", label="清洗触发")
        for t in triggers:
            plt.axvline(x=t, color="red", linestyle=":", alpha=0.35, linewidth=1)
    plt.axhline(THRESHOLD, color="#2ca02c", linestyle="--", linewidth=1.5, label=f"阈值 {THRESHOLD} kPa")
    plt.title(f"{target_col} 清洗预警效果（测试集）", fontsize=14, pad=12)
    plt.xlabel("时间序列索引（已按 TIME 排序）", fontsize=12)
    plt.ylabel("压差 (kPa)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.ticklabel_format(style="plain", axis="y")
    warn_path = os.path.join(OUT_PLOT_DIR, f"clean_alert_test_{safe_col_name}.png")
    plt.savefig(warn_path, dpi=150, bbox_inches="tight")
    print(f"[可视化] 清洗预警时间序列图已保存: {warn_path}")
    plt.show()

    # 可视化（取前 200 点）
    display_len = min(200, len(y_test))
    plt.figure(figsize=(12, 6))
    plt.plot(y_test[:display_len], label="真实观测值", color="#1f77b4", linewidth=2, alpha=0.8)
    plt.plot(y_pred[:display_len], label="XGBoost预测值", color="#d62728", linestyle="--", linewidth=2)
    plt.title(f"生产水处理系统压差动态预测 - {target_col} (RMSE={rmse:.2f}, MAE={mae:.2f})", fontsize=14)
    plt.xlabel("采样时间步", fontsize=12)
    plt.ylabel("压差 (kPa)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.ticklabel_format(style="plain", axis="y")
    save_path = os.path.join(OUT_PLOT_DIR, f"prediction_{safe_col_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[可视化] 图已保存: {save_path}")
    plt.show()

    # 测试集拟合散点图
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, color="#1f77b4", edgecolors="white", linewidths=0.6, label="测试集样本")
    min_v = min(y_test.min(), y_pred.min())
    max_v = max(y_test.max(), y_pred.max())
    plt.plot([min_v, max_v], [min_v, max_v], color="#d62728", linestyle="--", linewidth=2, label="y = x 参考线")
    plt.title(f"{target_col} 测试集拟合 (R={r:.3f}, R²={r2:.3f})", fontsize=13, pad=12)
    plt.xlabel("真实压差 y_true (kPa)", fontsize=12)
    plt.ylabel("预测压差 y_pred (kPa)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    scatter_path = os.path.join(OUT_PLOT_DIR, f"scatter_test_{safe_col_name}.png")
    plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
    print(f"[可视化] 散点拟合图已保存: {scatter_path}")
    plt.show()

    return model


if __name__ == "__main__":
    results_summary = {}
    for col in TARGET_COLUMNS:
        print(f"\n{'='*60}")
        print(f"开始处理 {col}")
        print(f"{'='*60}")
        model = run_model_pipeline(DATA_FILE, col)
        if model is not None:
            results_summary[col] = {"model": model}

    print(f"\n{'='*60}")
    print("三列处理完成！结果汇总：")
    print(f"{'='*60}")
    print(f"{'列名':<30} {'状态':<10}")
    print("-" * 40)
    for col in TARGET_COLUMNS:
        if col in results_summary:
            print(f"{col:<30} {'✓成功':<10}")
        else:
            print(f"{col:<30} {'✗失败':<10}")
