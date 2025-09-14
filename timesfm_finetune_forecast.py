#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
Water Level Forecasting with (Pseudo) Fine‑Tuned TimesFM

- Loads a CSV with date & target columns
- Splits Train/Valid/Test
- Initializes TimesFM (PyTorch checkpoint)
- Optional pretrained evaluation (MAE)
- Optional pseudo-finetune loop (template for real training)
- Rolling forecast (3-day horizon), plots, and hydrology metrics (NSE, KGE)
\"\"\"
import os
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless save
import matplotlib.pyplot as plt

import timesfm
import torch
from timesfm import data_loader

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def nse(observed: np.ndarray, forecasted: np.ndarray) -> float:
    observed = np.asarray(observed)
    forecasted = np.asarray(forecasted)
    denom = np.sum((observed - observed.mean()) ** 2)
    if denom == 0:
        return np.nan
    num = np.sum((observed - forecasted) ** 2)
    return 1.0 - (num / denom)

def kge(observed: np.ndarray, forecasted: np.ndarray) -> float:
    observed = np.asarray(observed)
    forecasted = np.asarray(forecasted)
    if observed.size < 2:
        return np.nan
    r = np.corrcoef(observed, forecasted)[0, 1]
    alpha = np.std(forecasted) / (np.std(observed) + 1e-12)
    beta = (np.mean(forecasted) + 1e-12) / (np.mean(observed) + 1e-12)
    return 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)

def plot_splits(df, train_idx, valid_idx, title, out_path):
    plt.figure(figsize=(12, 5))
    plt.plot(df['ds'], df['wl'], label="Water Level")
    plt.axvspan(df['ds'].iloc[0], df['ds'].iloc[train_idx-1], alpha=0.3, label="Train")
    plt.axvspan(df['ds'].iloc[train_idx], df['ds'].iloc[valid_idx-1], alpha=0.3, label="Valid")
    plt.axvspan(df['ds'].iloc[valid_idx], df['ds'].iloc[-1], alpha=0.3, label="Test")
    plt.xlabel("Date")
    plt.ylabel("Water Level")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

def load_and_prepare_series(data_file: str, date_col: str, target_col: str):
    df = pd.read_csv(data_file)
    if date_col not in df.columns or target_col not in df.columns:
        raise KeyError(f"Columns not found. Need: {date_col}, {target_col}")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    df['unique_id'] = 1
    df['ds'] = df[date_col]
    df['wl'] = df[target_col]
    series_df = df[['unique_id', 'ds', 'wl']].copy()
    return series_df

def build_timesfm_loader(tmp_csv_path: str, series_df: pd.DataFrame, hist_len: int, pred_len: int):
    series_df.to_csv(tmp_csv_path, index=False)
    dtl = data_loader.TimeSeriesdata(
        data_path=tmp_csv_path,
        datetime_col="ds",
        ts_cols=np.array(["wl"]),
        num_cov_cols=None,
        cat_cov_cols=None,
        train_range=[0, int(len(series_df) * 0.8)],
        val_range=[int(len(series_df) * 0.8), int(len(series_df) * 0.9)],
        test_range=[int(len(series_df) * 0.9), len(series_df)],
        hist_len=hist_len,
        pred_len=pred_len,
        batch_size=1,     # single time series
        freq="d",
        normalize=True,
        permute=True,
    )
    return dtl

def init_timesfm(horizon: int, context_len: int):
    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="gpu" if torch.cuda.is_available() else "cpu",
            per_core_batch_size=16,
            horizon_len=horizon,
            context_len=context_len,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
        ),
    )
    return tfm

def evaluate_pretrained_mae(tfm, dtl) -> float:
    maes = []
    # shift=1 evaluates one-step ahead with provided loader semantics
    for batch in dtl.tf_dataset(mode="test", shift=1).as_numpy_iterator():
        forecasts, _ = tfm.forecast(
            list(batch[0]),
            [0] * len(batch[0]),
            normalize=True
        )
        y_true = batch[3]  # adapter expects target at index 3
        maes.append(np.mean(np.abs(forecasts - y_true)))
    mae = float(np.mean(maes)) if maes else float("nan")
    return mae

def pseudo_finetune(tfm, dtl, epochs: int, patience: int, model_out: str | None):
    best_val = float("inf")
    wait = 0
    for ep in range(1, epochs + 1):
        # "Training" loop — placeholder (no weight updates by default)
        train_maes = []
        for batch in dtl.tf_dataset(mode="train", shift=1).as_numpy_iterator():
            forecasts, _ = tfm.forecast(
                list(batch[0]),
                [0] * len(batch[0]),
                normalize=True
            )
            y_true = batch[3]
            train_maes.append(np.mean(np.abs(forecasts - y_true)))
        train_mae = float(np.mean(train_maes)) if train_maes else float("nan")
        print(f"Epoch {ep}: Train MAE={train_mae:.5f}")

        # "Validation" loop — monitor early stopping
        val_maes = []
        for batch in dtl.tf_dataset(mode="val", shift=1).as_numpy_iterator():
            forecasts, _ = tfm.forecast(
                list(batch[0]),
                [0] * len(batch[0]),
                normalize=True
            )
            y_true = batch[3]
            val_maes.append(np.mean(np.abs(forecasts - y_true)))
        val_mae = float(np.mean(val_maes)) if val_maes else float("nan")
        print(f"          Val   MAE={val_mae:.5f}")

        if val_mae < best_val:
            best_val = val_mae
            wait = 0
            if model_out:
                try:
                    with open(model_out, "wb") as f:
                        pickle.dump(tfm, f)
                    print(f"Saved model to {model_out}")
                except Exception as e:
                    print(f"Warning: could not save model: {e}")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

def rolling_forecast_and_metrics(tfm, series_df: pd.DataFrame, horizon: int, plots_dir: str, gauge_name: str):
    # Build splits for plotting
    n = len(series_df)
    train_idx = int(0.8 * n)
    valid_idx = int(0.9 * n)
    plot_splits(series_df, train_idx, valid_idx,
                title=f"Measured Water Level — {gauge_name} (Train/Valid/Test)",
                out_path=os.path.join(plots_dir, "split_overview.png"))

    train_df = series_df.iloc[:train_idx].copy()
    test_df  = series_df.iloc[valid_idx:].copy()

    # Rolling 3-day ahead (or horizon) forecast
    results = []
    steps = len(test_df) - (horizon - 1)
    if steps <= 0:
        raise ValueError("Not enough test points for the requested horizon.")

    for i in range(steps):
        fc_df = tfm.forecast_on_df(
            inputs=train_df,
            freq="D",
            value_name="wl",
            num_jobs=-1,
        )

        # Extract predictions
        if "timesfm" in fc_df.columns:
            preds = np.array(fc_df["timesfm"].iloc[0]).reshape(-1)
        elif "yhat" in fc_df.columns:
            preds = np.array(fc_df["yhat"].iloc[0]).reshape(-1)
        else:
            preds = np.array(fc_df.iloc[0].tolist()[0]).reshape(-1)

        if preds.shape[0] < horizon:
            raise RuntimeError("TimesFM did not return the expected horizon length.")

        results.append({
            "date": test_df["ds"].iloc[i],
            "h1": float(preds[0]),
            "h2": float(preds[1]) if horizon >= 2 else np.nan,
            "h3": float(preds[2]) if horizon >= 3 else np.nan,
            "obs": float(test_df["wl"].iloc[i]),
        })

        # Online update
        train_df = pd.concat([train_df, test_df.iloc[[i]]], ignore_index=True)

    results_df = pd.DataFrame(results)

    # Align forward by one step (visual)
    for col in ["h1","h2","h3"]:
        if col in results_df.columns:
            results_df[col] = results_df[col].shift(-1)
    results_df = results_df.iloc[:-1].reset_index(drop=True)

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(results_df["date"], results_df["obs"], label="Observed")
    if "h1" in results_df:
        plt.plot(results_df["date"], results_df["h1"], label="Forecast (Day 1)")
    if "h2" in results_df and not results_df["h2"].isna().all():
        plt.plot(results_df["date"], results_df["h2"], label="Forecast (Day 2)")
    if "h3" in results_df and not results_df["h3"].isna().all():
        plt.plot(results_df["date"], results_df["h3"], label="Forecast (Day 3)")
    plt.xlabel("Date")
    plt.ylabel("Water Level")
    plt.title(f"TimesFM Rolling Forecast (3-day ahead) vs Observed — {gauge_name}")
    plt.legend()
    plt.tight_layout()
    ensure_dir(plots_dir)
    plt.savefig(os.path.join(plots_dir, "forecast_vs_obs.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Metrics per horizon
    obs = results_df["obs"].values
    for col in ["h1","h2","h3"]:
        if col in results_df.columns:
            preds = results_df[col].values
            mask = ~np.isnan(preds)
            if mask.any():
                print(f"{col.upper()} NSE: {nse(obs[mask], preds[mask]):.4f}")
                print(f"{col.upper()} KGE: {kge(obs[mask], preds[mask]):.4f}")

def main():
    parser = argparse.ArgumentParser(description="Water Level Forecasting with (Pseudo) Fine‑Tuned TimesFM")
    parser.add_argument("--data_file", type=str, required=True, help="Path to CSV with date & target cols.")
    parser.add_argument("--date_col", type=str, default="date", help="Date column name (default: date).")
    parser.add_argument("--target_col", type=str, default="wl", help="Target column name (default: wl).")
    parser.add_argument("--gauge_name", type=str, default="Gauge", help="Name used in plot titles.")
    parser.add_argument("--horizon", type=int, default=3, help="Forecast horizon (default: 3).")
    parser.add_argument("--context_len", type=int, default=352, help="TimesFM context length (default: 352).")
    parser.add_argument("--hist_len", type=int, default=14, help="Historical length for loader (default: 14).")
    parser.add_argument("--train_frac", type=float, default=0.8, help="Train fraction (default: 0.8).")
    parser.add_argument("--valid_frac", type=float, default=0.1, help="Validation fraction (default: 0.1).")
    parser.add_argument("--plots_dir", type=str, default="plots", help="Output plots directory.")
    parser.add_argument("--evaluate_pretrained", action="store_true", help="Evaluate pretrained model MAE on test.")
    parser.add_argument("--pseudo_finetune", action="store_true", help="Run pseudo-finetune loop (no weight updates).")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs for pseudo-finetune.")
    parser.add_argument("--patience", type=int, default=5, help="Early-stopping patience.")
    parser.add_argument("--model_out", type=str, default="timesfm_finetuned.pkl", help="Pickle path to save model.")
    args = parser.parse_args()

    ensure_dir(args.plots_dir)

    # 1) Load series
    series_df = load_and_prepare_series(args.data_file, args.date_col, args.target_col)

    # 2) Build loader CSV (for dtl API)
    tmp_csv_path = os.path.join(".", "_timesfm_series_tmp.csv")
    dtl = build_timesfm_loader(tmp_csv_path, series_df, hist_len=args.hist_len, pred_len=args.horizon)

    # 3) Init model
    tfm = init_timesfm(horizon=args.horizon, context_len=args.context_len)

    # 4) Optional: evaluate pretrained
    if args.evaluate_pretrained:
        mae = evaluate_pretrained_mae(tfm, dtl)
        print(f"Pretrained Model Test MAE: {mae:.5f}")

    # 5) Optional: pseudo-finetune
    if args.pseudo_finetune:
        pseudo_finetune(tfm, dtl, epochs=args.epochs, patience=args.patience, model_out=args.model_out)

        # Try to reload the saved model, if any
        if args.model_out and os.path.exists(args.model_out):
            try:
                with open(args.model_out, "rb") as f:
                    tfm = pickle.load(f)
                print(f"Reloaded model from {args.model_out}")
            except Exception as e:
                print(f"Warning: could not reload model: {e}")

    # 6) Rolling forecast + metrics
    rolling_forecast_and_metrics(tfm, series_df, horizon=args.horizon, plots_dir=args.plots_dir, gauge_name=args.gauge_name)

if __name__ == "__main__":
    main()
