# -*- coding: utf-8 -*-
"""Train LightGBM to predict Taiwan stock / ETF 5-day direction.

功能：
  1. 讀 all_price.csv，或掃描資料夾內所有 *_price.csv
  2. 建立技術面特徵，只使用當日與過去資料，避免資料洩漏
  3. 建立未來 5 日報酬與漲跌 target
  4. 依日期切 train / valid / test，不用隨機切分
  5. 訓練 LightGBM binary classifier
  6. 輸出：
      - prediction_5d_lightgbm.csv       最新一日每檔未來 5 日看漲機率
      - backtest_report.csv              train / valid / test 指標
      - backtest_daily_topN.csv          測試期間每日挑前 N 檔的 5 日報酬
      - feature_importance_lightgbm.csv  特徵重要度
      - lightgbm_5d_model.txt            LightGBM 模型

安裝：
    pip install lightgbm pandas numpy

基本用法：
    python train_lightgbm_5d.py --input all_price.csv

沒有 all_price.csv，直接掃描目前資料夾：
    python train_lightgbm_5d.py --csv-dir .

大量資料建議：
    python train_lightgbm_5d.py --input all_price.csv --top-n 50 --num-boost-round 800

注意：
    這不是投資建議。這只是基於歷史價量資料的研究/篩選工具。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass


EXCLUDE_PREFIXES = (
    "failed_",
    "prediction_",
    "backtest_",
    "feature_importance_",
    "stocks_meta",
)


BASE_FEATURES = [
    "ret_1",
    "ret_2",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_20",
    "volatility_5",
    "volatility_10",
    "volatility_20",
    "ma5_gap",
    "ma10_gap",
    "ma20_gap",
    "ma60_gap",
    "ma5_ma20_gap",
    "ma20_ma60_gap",
    "ma5_slope_5",
    "ma20_slope_10",
    "volume_ratio_5",
    "volume_ratio_20",
    "volume_trend_5_20",
    "turnover_ratio_5",
    "candle_return",
    "intraday_range",
    "close_position",
    "rsi_14",
    "bb_percent_b",
    "dist_20d_high",
    "dist_20d_low",
    "weekday",
]


def print_install_help() -> None:
    print("缺少 lightgbm，請先安裝：")
    print("    pip install lightgbm")
    print()
    print("如果你用的是 Anaconda：")
    print("    conda install -c conda-forge lightgbm")


def safe_to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def stock_id_from_filename(path: Path) -> str:
    if path.name.endswith("_price.csv"):
        return path.name[: -len("_price.csv")]
    return path.stem


def should_skip_csv(path: Path) -> bool:
    name = path.name.lower()
    if not name.endswith("_price.csv"):
        return True
    if name == "all_price.csv":
        return True
    return any(name.startswith(prefix) for prefix in EXCLUDE_PREFIXES)


def normalize_columns(df: pd.DataFrame, source_path: Optional[Path] = None) -> pd.DataFrame:
    df = df.copy()

    rename_map = {}
    for col in df.columns:
        lower = str(col).strip().lower()
        if lower == "stockid":
            rename_map[col] = "stock_id"
        elif lower == "trading_volume":
            rename_map[col] = "Trading_Volume"
        elif lower == "trading_money":
            rename_map[col] = "Trading_money"
        elif lower == "trading_turnover":
            rename_map[col] = "Trading_turnover"
        elif lower == "max":
            rename_map[col] = "high"
        elif lower == "min":
            rename_map[col] = "low"

    if rename_map:
        df = df.rename(columns=rename_map)

    if "date" not in df.columns:
        raise ValueError("缺少 date 欄位")
    if "close" not in df.columns:
        raise ValueError("缺少 close 欄位")
    if "stock_id" not in df.columns:
        if source_path is None:
            raise ValueError("缺少 stock_id 欄位，且沒有檔名可推斷")
        df["stock_id"] = stock_id_from_filename(source_path)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["stock_id"] = df["stock_id"].astype(str).str.strip()

    # 某些 CSV stock_id 可能被讀成 2330.0
    df["stock_id"] = df["stock_id"].str.replace(r"\.0$", "", regex=True)

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "spread",
        "Trading_Volume",
        "Trading_money",
        "Trading_turnover",
        "ma5",
        "ma20",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = safe_to_numeric(df[col])

    if "open" not in df.columns:
        df["open"] = np.nan
    if "high" not in df.columns:
        df["high"] = np.nan
    if "low" not in df.columns:
        df["low"] = np.nan
    if "Trading_Volume" not in df.columns:
        df["Trading_Volume"] = np.nan
    if "Trading_turnover" not in df.columns:
        df["Trading_turnover"] = np.nan

    df = df.dropna(subset=["date", "stock_id", "close"])
    df = df.drop_duplicates(subset=["stock_id", "date"], keep="last")
    return df


def load_prices(input_file: Optional[str], csv_dir: str) -> pd.DataFrame:
    if input_file:
        path = Path(input_file)
        if not path.exists():
            raise FileNotFoundError(f"找不到輸入檔：{path}")
        print(f"讀取合併檔：{path}")
        return normalize_columns(pd.read_csv(path, dtype={"stock_id": str}), source_path=None)

    folder = Path(csv_dir)
    if not folder.exists():
        raise FileNotFoundError(f"找不到資料夾：{folder}")

    frames = []
    paths = sorted(folder.glob("*_price.csv"))
    for i, path in enumerate(paths, start=1):
        if should_skip_csv(path):
            continue
        try:
            one = pd.read_csv(path, dtype={"stock_id": str})
            frames.append(normalize_columns(one, source_path=path))
        except Exception as exc:
            print(f"略過 {path.name}: {exc}")

        if i % 500 == 0:
            print(f"  已掃描 {i}/{len(paths)} 個 CSV")

    if not frames:
        raise RuntimeError(f"在 {folder.resolve()} 找不到可用的 *_price.csv")

    return pd.concat(frames, ignore_index=True)


def calc_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()

    rs = gain / loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.where(~((loss == 0) & (gain > 0)), 100.0)
    rsi = rsi.where(~((gain == 0) & (loss > 0)), 0.0)
    return rsi


def add_features_one_stock(one: pd.DataFrame, horizon: int, target_return: float) -> pd.DataFrame:
    one = one.sort_values("date").copy()
    one["bar_index"] = np.arange(len(one))

    close = one["close"]
    volume = one["Trading_Volume"]
    turnover = one["Trading_turnover"]

    for n in [1, 2, 3, 5, 10, 20]:
        one[f"ret_{n}"] = close.pct_change(n)

    for n in [5, 10, 20]:
        one[f"volatility_{n}"] = close.pct_change().rolling(n, min_periods=n).std()

    ma5 = close.rolling(5, min_periods=5).mean()
    ma10 = close.rolling(10, min_periods=10).mean()
    ma20 = close.rolling(20, min_periods=20).mean()
    ma60 = close.rolling(60, min_periods=60).mean()

    one["ma5_gap"] = close / ma5 - 1.0
    one["ma10_gap"] = close / ma10 - 1.0
    one["ma20_gap"] = close / ma20 - 1.0
    one["ma60_gap"] = close / ma60 - 1.0
    one["ma5_ma20_gap"] = ma5 / ma20 - 1.0
    one["ma20_ma60_gap"] = ma20 / ma60 - 1.0
    one["ma5_slope_5"] = ma5 / ma5.shift(5) - 1.0
    one["ma20_slope_10"] = ma20 / ma20.shift(10) - 1.0

    vol5 = volume.rolling(5, min_periods=5).mean()
    vol20 = volume.rolling(20, min_periods=20).mean()
    one["volume_ratio_5"] = volume / vol5
    one["volume_ratio_20"] = volume / vol20
    one["volume_trend_5_20"] = vol5 / vol20 - 1.0

    turn5 = turnover.rolling(5, min_periods=5).mean()
    one["turnover_ratio_5"] = turnover / turn5

    one["candle_return"] = np.where(one["open"] != 0, close / one["open"] - 1.0, np.nan)
    one["intraday_range"] = np.where(close != 0, (one["high"] - one["low"]) / close, np.nan)
    one["close_position"] = np.where(
        (one["high"] - one["low"]) != 0,
        (close - one["low"]) / (one["high"] - one["low"]),
        np.nan,
    )

    one["rsi_14"] = calc_rsi(close, 14)

    rolling_mean_20 = close.rolling(20, min_periods=20).mean()
    rolling_std_20 = close.rolling(20, min_periods=20).std()
    bb_upper = rolling_mean_20 + 2 * rolling_std_20
    bb_lower = rolling_mean_20 - 2 * rolling_std_20
    one["bb_percent_b"] = (close - bb_lower) / (bb_upper - bb_lower)

    high20 = close.rolling(20, min_periods=20).max()
    low20 = close.rolling(20, min_periods=20).min()
    one["dist_20d_high"] = close / high20 - 1.0
    one["dist_20d_low"] = close / low20 - 1.0

    one["weekday"] = one["date"].dt.weekday

    one["future_close_5d"] = close.shift(-horizon)
    one["future_return_5d"] = one["future_close_5d"] / close - 1.0
    one["target_5d_up"] = (one["future_return_5d"] > target_return).astype(float)
    one.loc[one["future_return_5d"].isna(), "target_5d_up"] = np.nan

    return one


def build_feature_table(prices: pd.DataFrame, horizon: int, target_return: float) -> pd.DataFrame:
    prices = prices.sort_values(["stock_id", "date"]).copy()

    frames = []
    stock_ids = prices["stock_id"].dropna().astype(str).unique()
    for i, stock_id in enumerate(stock_ids, start=1):
        one = prices[prices["stock_id"].astype(str) == str(stock_id)]
        frames.append(add_features_one_stock(one, horizon=horizon, target_return=target_return))
        if i % 500 == 0:
            print(f"  特徵處理 {i}/{len(stock_ids)} 檔")

    return pd.concat(frames, ignore_index=True)


def clean_feature_values(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    df = df.copy()

    # 對極端值做 winsorize，避免單日異常值支配模型。
    for col in feature_cols:
        if col not in df.columns:
            continue
        df[col] = safe_to_numeric(df[col])
        finite = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = finite

        # 大部分技術特徵是比例，限制在合理區間。
        if col not in ["weekday", "rsi_14", "close_position", "bb_percent_b"]:
            df[col] = df[col].clip(-5.0, 5.0)

    return df


def get_split_dates(
    dates: Iterable[pd.Timestamp],
    train_ratio: float,
    valid_ratio: float,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    unique_dates = pd.Series(pd.to_datetime(list(dates))).dropna().drop_duplicates().sort_values().reset_index(drop=True)
    if len(unique_dates) < 30:
        raise RuntimeError(f"可用日期太少：{len(unique_dates)}，建議拉長資料區間")

    train_idx = int(len(unique_dates) * train_ratio)
    valid_idx = int(len(unique_dates) * (train_ratio + valid_ratio))

    train_idx = max(1, min(train_idx, len(unique_dates) - 3))
    valid_idx = max(train_idx + 1, min(valid_idx, len(unique_dates) - 2))

    return unique_dates.iloc[train_idx], unique_dates.iloc[valid_idx]


def make_splits(
    df: pd.DataFrame,
    train_ratio: float,
    valid_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    train_end, valid_end = get_split_dates(df["date"], train_ratio, valid_ratio)

    train = df[df["date"] <= train_end].copy()
    valid = df[(df["date"] > train_end) & (df["date"] <= valid_end)].copy()
    test = df[df["date"] > valid_end].copy()

    if train.empty or valid.empty or test.empty:
        raise RuntimeError(
            f"切分後資料不足：train={len(train)}, valid={len(valid)}, test={len(test)}。"
            "請調整 --train-ratio / --valid-ratio 或拉長資料區間。"
        )

    return train, valid, test, train_end, valid_end


def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_score = np.asarray(y_score).astype(float)

    mask = ~np.isnan(y_true) & ~np.isnan(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = pd.Series(y_score).rank(method="average").to_numpy()
    sum_pos_ranks = ranks[y_true == 1].sum()
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def classification_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(float)
    prob = np.asarray(prob).astype(float)

    mask = ~np.isnan(y_true) & ~np.isnan(prob)
    y_true = y_true[mask]
    prob = prob[mask]

    if len(y_true) == 0:
        return {
            "samples": 0,
            "positive_rate": float("nan"),
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "auc": float("nan"),
        }

    pred = (prob >= threshold).astype(int)
    y = y_true.astype(int)

    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    accuracy = (tp + tn) / len(y)
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision == precision and recall == recall and (precision + recall) else float("nan")

    return {
        "samples": int(len(y)),
        "positive_rate": float(y.mean()),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": auc_score(y, prob),
    }


def train_lightgbm(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_cols: List[str],
    args: argparse.Namespace,
):
    try:
        import lightgbm as lgb
    except ModuleNotFoundError:
        print_install_help()
        raise

    X_train = train[feature_cols]
    y_train = train["target_5d_up"].astype(int)
    X_valid = valid[feature_cols]
    y_valid = valid["target_5d_up"].astype(int)

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols, free_raw_data=False)
    valid_data = lgb.Dataset(X_valid, label=y_valid, feature_name=feature_cols, reference=train_data, free_raw_data=False)

    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "boosting_type": "gbdt",
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": args.bagging_freq,
        "lambda_l1": args.lambda_l1,
        "lambda_l2": args.lambda_l2,
        "seed": args.seed,
        "feature_pre_filter": False,
        "verbosity": -1,
    }

    if args.is_unbalance:
        params["is_unbalance"] = True

    callbacks = [
        lgb.log_evaluation(period=args.log_period),
        lgb.early_stopping(stopping_rounds=args.early_stopping_rounds, verbose=True),
    ]

    model = lgb.train(
        params=params,
        train_set=train_data,
        num_boost_round=args.num_boost_round,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )

    return model


def evaluate_splits(
    model,
    splits: Dict[str, pd.DataFrame],
    feature_cols: List[str],
    threshold: float,
) -> pd.DataFrame:
    rows = []
    for split_name, part in splits.items():
        if part.empty:
            continue
        prob = model.predict(part[feature_cols], num_iteration=model.best_iteration)
        metrics = classification_metrics(part["target_5d_up"].to_numpy(), prob, threshold=threshold)
        metrics["split"] = split_name
        rows.append(metrics)
    return pd.DataFrame(rows)[["split", "samples", "positive_rate", "accuracy", "precision", "recall", "f1", "auc"]]


def topn_backtest(
    model,
    test: pd.DataFrame,
    feature_cols: List[str],
    top_n: int,
) -> pd.DataFrame:
    if test.empty:
        return pd.DataFrame()

    test = test.copy()
    test["prob_up"] = model.predict(test[feature_cols], num_iteration=model.best_iteration)

    rows = []
    for date, group in test.groupby("date"):
        selected = group.sort_values("prob_up", ascending=False).head(top_n)
        if selected.empty:
            continue
        rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "selected_count": int(len(selected)),
                "avg_prob_up": float(selected["prob_up"].mean()),
                "avg_future_5d_return": float(selected["future_return_5d"].mean()),
                "median_future_5d_return": float(selected["future_return_5d"].median()),
                "hit_rate": float((selected["future_return_5d"] > 0).mean()),
                "stock_ids": ",".join(selected["stock_id"].astype(str).tolist()),
            }
        )

    return pd.DataFrame(rows)


def latest_prediction(
    model,
    feature_table: pd.DataFrame,
    feature_cols: List[str],
    min_history: int,
) -> pd.DataFrame:
    latest_rows = []
    for stock_id, one in feature_table.sort_values("date").groupby("stock_id"):
        valid_feature_rows = one[one["bar_index"] >= min_history].copy()
        if valid_feature_rows.empty:
            continue
        latest_rows.append(valid_feature_rows.iloc[-1])

    if not latest_rows:
        return pd.DataFrame()

    latest = pd.DataFrame(latest_rows).copy()
    latest["prob_up_5d"] = model.predict(latest[feature_cols], num_iteration=model.best_iteration)
    latest["prob_down_5d"] = 1.0 - latest["prob_up_5d"]
    latest["prediction_5d"] = np.where(latest["prob_up_5d"] >= 0.5, "漲", "跌")

    out_cols = [
        "stock_id",
        "date",
        "close",
        "prediction_5d",
        "prob_up_5d",
        "prob_down_5d",
        "ret_1",
        "ret_5",
        "ret_20",
        "ma5_gap",
        "ma20_gap",
        "rsi_14",
        "volume_ratio_5",
        "volume_ratio_20",
    ]

    for col in out_cols:
        if col not in latest.columns:
            latest[col] = np.nan

    result = latest[out_cols].copy()
    result["date"] = pd.to_datetime(result["date"]).dt.strftime("%Y-%m-%d")
    result["prob_up_5d"] = result["prob_up_5d"].round(6)
    result["prob_down_5d"] = result["prob_down_5d"].round(6)
    result = result.sort_values("prob_up_5d", ascending=False)
    return result


def save_feature_importance(model, feature_cols: List[str], output_dir: Path) -> None:
    importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_gain": model.feature_importance(importance_type="gain"),
            "importance_split": model.feature_importance(importance_type="split"),
        }
    )
    importance = importance.sort_values("importance_gain", ascending=False)
    importance.to_csv(output_dir / "feature_importance_lightgbm.csv", index=False, encoding="utf-8-sig")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train LightGBM to predict 5-day Taiwan stock direction.")
    parser.add_argument("--input", help="合併檔 all_price.csv。若未指定，會掃描 --csv-dir 的 *_price.csv。")
    parser.add_argument("--csv-dir", default=".", help="CSV 資料夾，預設目前資料夾。")
    parser.add_argument("--output-dir", default=".", help="輸出資料夾，預設目前資料夾。")

    parser.add_argument("--horizon", type=int, default=5, help="預測未來幾個交易日，預設 5。")
    parser.add_argument("--target-return", type=float, default=0.0, help="未來報酬大於此值才算漲，預設 0。")
    parser.add_argument("--min-history", type=int, default=60, help="每檔至少有幾筆歷史才納入訓練，預設 60。")

    parser.add_argument("--train-ratio", type=float, default=0.70, help="依日期排序後，訓練集比例，預設 0.70。")
    parser.add_argument("--valid-ratio", type=float, default=0.15, help="驗證集比例，預設 0.15。")
    parser.add_argument("--threshold", type=float, default=0.50, help="分類門檻，預設 0.50。")
    parser.add_argument("--top-n", type=int, default=50, help="測試期間每日挑看漲機率前 N 檔做回測，預設 50。")

    parser.add_argument("--num-boost-round", type=int, default=800)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--min-data-in-leaf", type=int, default=80)
    parser.add_argument("--feature-fraction", type=float, default=0.85)
    parser.add_argument("--bagging-fraction", type=float, default=0.85)
    parser.add_argument("--bagging-freq", type=int, default=1)
    parser.add_argument("--lambda-l1", type=float, default=0.0)
    parser.add_argument("--lambda-l2", type=float, default=1.0)
    parser.add_argument("--is-unbalance", action="store_true", help="若漲跌比例不均，讓 LightGBM 自動處理類別不平衡。")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-period", type=int, default=50)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.train_ratio <= 0 or args.valid_ratio <= 0 or args.train_ratio + args.valid_ratio >= 0.95:
        raise ValueError("--train-ratio + --valid-ratio 必須小於 0.95，且兩者都要大於 0")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("讀取資料...")
    prices = load_prices(args.input, args.csv_dir)
    prices = prices.sort_values(["stock_id", "date"]).reset_index(drop=True)

    print(f"原始資料筆數: {len(prices):,}")
    print(f"股票/ETF 數量: {prices['stock_id'].nunique():,}")
    print(f"日期區間: {prices['date'].min().date()} ~ {prices['date'].max().date()}")

    print()
    print("建立特徵與 5 日 target...")
    feature_table = build_feature_table(prices, horizon=args.horizon, target_return=args.target_return)

    feature_cols = BASE_FEATURES[:]
    feature_table = clean_feature_values(feature_table, feature_cols)

    trainable = feature_table.dropna(subset=["target_5d_up", "future_return_5d"]).copy()
    trainable = trainable[trainable["bar_index"] >= args.min_history].copy()

    # LightGBM 可以處理 feature NaN，所以不用 dropna(feature_cols)。
    if trainable.empty:
        raise RuntimeError("可訓練資料為空。請拉長日期區間或降低 --min-history。")

    print(f"可訓練資料筆數: {len(trainable):,}")
    print(f"target 漲的比例: {trainable['target_5d_up'].mean():.4f}")

    train, valid, test, train_end, valid_end = make_splits(
        trainable,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
    )

    print()
    print("時間切分：")
    print(f"  train: <= {train_end.date()}，筆數 {len(train):,}")
    print(f"  valid: > {train_end.date()} 且 <= {valid_end.date()}，筆數 {len(valid):,}")
    print(f"  test : > {valid_end.date()}，筆數 {len(test):,}")

    print()
    print("訓練 LightGBM...")
    model = train_lightgbm(train, valid, feature_cols, args)

    print()
    print("評估 train / valid / test...")
    report = evaluate_splits(
        model,
        splits={"train": train, "valid": valid, "test": test},
        feature_cols=feature_cols,
        threshold=args.threshold,
    )
    report_path = output_dir / "backtest_report.csv"
    report.to_csv(report_path, index=False, encoding="utf-8-sig")
    print(report.to_string(index=False))
    print(f"已輸出: {report_path.resolve()}")

    print()
    print(f"測試期間每日挑看漲機率前 {args.top_n} 檔...")
    daily_topn = topn_backtest(model, test, feature_cols, top_n=args.top_n)
    daily_topn_path = output_dir / "backtest_daily_topN.csv"
    daily_topn.to_csv(daily_topn_path, index=False, encoding="utf-8-sig")
    print(f"已輸出: {daily_topn_path.resolve()}")

    if not daily_topn.empty:
        summary = {
            "top_n": args.top_n,
            "days": int(len(daily_topn)),
            "avg_5d_return": float(daily_topn["avg_future_5d_return"].mean()),
            "median_5d_return": float(daily_topn["avg_future_5d_return"].median()),
            "positive_day_rate": float((daily_topn["avg_future_5d_return"] > 0).mean()),
            "avg_hit_rate": float(daily_topn["hit_rate"].mean()),
        }
        summary_path = output_dir / "backtest_topN_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print("TopN 摘要:")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"已輸出: {summary_path.resolve()}")

    print()
    print("輸出最新一日 5 日預測...")
    pred = latest_prediction(model, feature_table, feature_cols, min_history=args.min_history)
    pred_path = output_dir / "prediction_5d_lightgbm.csv"
    pred.to_csv(pred_path, index=False, encoding="utf-8-sig")
    print(f"已輸出: {pred_path.resolve()}")

    if not pred.empty:
        print()
        print("看漲機率最高前 20 檔：")
        print(pred.head(20).to_string(index=False))

    save_feature_importance(model, feature_cols, output_dir=output_dir)
    print(f"已輸出: {(output_dir / 'feature_importance_lightgbm.csv').resolve()}")

    model_path = output_dir / "lightgbm_5d_model.txt"
    model.save_model(str(model_path))
    print(f"已輸出: {model_path.resolve()}")

    print()
    print("完成。提醒：請把 backtest_report.csv 與 backtest_daily_topN.csv 當成評估重點，不要只看最新預測。")


if __name__ == "__main__":
    main()
