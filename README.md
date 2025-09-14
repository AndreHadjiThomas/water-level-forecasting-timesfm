# Water Level Forecasting with (Pseudo) Fine‑Tuned TimesFM

This repository demonstrates a **TimesFM**–based pipeline for water‑level forecasting with a **rolling 3‑day horizon**, plots, and hydrology metrics (**NSE** and **KGE**). It also includes a *pseudo fine‑tuning* loop over the training data to mirror the notebook flow. (TimesFM is commonly used for inference; if you have a trainable variant, replace the placeholder loop with weight updates.)

## What it does
- Loads a CSV (date + target column)
- Splits into **Train / Valid / Test** (defaults: 80/10/10)
- Initializes **TimesFM** with a pretrained Hugging Face checkpoint
- (Optional) Evaluates the pretrained model on test MAE
- (Optional) Runs a **pseudo‑finetune** loop (no gradient update by default; acts as a template)
- Runs a **rolling 3‑day‑ahead** forecast and plots **observed vs. h1/h2/h3**
- Prints **NSE** and **KGE** for each horizon
- Saves outputs to `plots/`

> **Note:** The script uses the PyTorch checkpoint `google/timesfm-1.0-200m-pytorch` and requires the `timesfm` Python package.

## Data expectations
Your CSV should contain at least:
- a date column (default: `date`)
- a target column with observed water level (default: `wl`)

Adjust names via CLI flags.

## Quickstart
```bash
# (optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# install deps
pip install -r requirements.txt

# run (with defaults)
python timesfm_finetune_forecast.py   --data_file "/path/to/your.csv"   --date_col "date"   --target_col "wl"   --gauge_name "MyGauge"   --horizon 3   --context_len 352   --hist_len 14   --train_frac 0.8   --valid_frac 0.1   --plots_dir "plots"   --evaluate_pretrained   --pseudo_finetune   --epochs 5   --patience 5   --model_out "timesfm_finetuned.pkl"
```

This will generate:
- `plots/split_overview.png` — shaded Train/Valid/Test ranges
- `plots/forecast_vs_obs.png` — observed vs. h1/h2/h3
- Console metrics: **NSE** / **KGE** for h1/h2/h3
- (Optional) a pickled model file `timesfm_finetuned.pkl`

## CLI options
Run `python timesfm_finetune_forecast.py -h` for all flags. Key ones:
- `--data_file` (str): path to CSV
- `--date_col` (str, default `date`): date column
- `--target_col` (str, default `wl`): target column
- `--gauge_name` (str): for plot titles
- `--horizon` (int, default 3): forecast horizon
- `--context_len` (int, default 352): TimesFM context length
- `--hist_len` (int, default 14): for loader (training context)
- `--train_frac` / `--valid_frac` (floats): split fractions
- `--plots_dir` (str): folder for PNGs
- `--evaluate_pretrained` (flag): print MAE on test set, no updates
- `--pseudo_finetune` (flag): run a placeholder training loop (no weight updates)
- `--epochs` / `--patience`: loop & early‑stopping parameters
- `--model_out` (str): pickle path

## Notes
- If you have a *trainable* TimesFM variant, insert your **optimizer / backprop** inside the training loop where noted.
- The rolling forecast uses `TimesFm.forecast_on_df`. If your `timesfm` version returns different column names, adjust the extraction block accordingly.
- For GPU, install a CUDA‑enabled PyTorch per https://pytorch.org/get-started/locally/

## License
MIT
