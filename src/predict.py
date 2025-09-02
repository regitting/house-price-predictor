import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import load

def load_expected_columns(meta_path="models/metadata.json"):
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta["expected_columns"]

def align_columns(df: pd.DataFrame, expected_cols):
    """Ensure df has exactly the expected columns (order, add missing as NaN, drop extras)."""
    missing = [c for c in expected_cols if c not in df.columns]
    extras = [c for c in df.columns if c not in expected_cols]

    if missing:
        for c in missing:
            df[c] = np.nan
    if extras:
        df = df.drop(columns=extras)

    # reorder
    return df[expected_cols], missing, extras

def main():
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    default_out = Path("predictions") / f"predictions_{ts}.csv"

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to CSV containing raw features")
    ap.add_argument("--out", default=str(default_out), help="Path to output predictions CSV")
    ap.add_argument("--model", default="models/house_price_pipeline.joblib")
    ap.add_argument("--meta", default="models/metadata.json", help="Path to metadata.json with expected columns")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # load
    X_raw = pd.read_csv(args.csv)
    expected_cols = load_expected_columns(args.meta)
    X_aligned, missing, extras = align_columns(X_raw.copy(), expected_cols)

    if missing:
        print(f"[warn] Added missing columns (filled NaN): {missing}")
    if extras:
        print(f"[warn] Dropped extra columns: {extras}")

    pipe = load(args.model)

    # model was trained on log target; pipeline outputs log-price predictions
    log_preds = pipe.predict(X_aligned)
    preds = np.expm1(log_preds)

    out = X_raw.copy()
    out["PredictedPriceUSD"] = preds
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with {len(out)} rows")

if __name__ == "__main__":
    main()
