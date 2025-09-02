import json
import numpy as np
from joblib import dump
from pathlib import Path
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor

from data_loader import load_california
from features import split_columns, build_preprocessor

def main():
    X, y = load_california(as_log_target=True)

    num_cols, cat_cols = split_columns(X)
    pre = build_preprocessor(num_cols, cat_cols)

    model = LGBMRegressor(
        n_estimators=800, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42
    )
    pipe = Pipeline([("pre", pre), ("model", model)])

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_rmse = -cross_val_score(pipe, X, y, scoring="neg_root_mean_squared_error", cv=cv)
    print(f"CV RMSE (log target): mean={cv_rmse.mean():.4f}, std={cv_rmse.std():.4f}")

    pipe.fit(X, y)

    Path("models").mkdir(exist_ok=True)
    dump(pipe, "models/house_price_pipeline.joblib")
    print("Saved model to models/house_price_pipeline.joblib")

    # save expected raw input columns from X (before preprocessing)
    meta = {
        "expected_columns": list(X.columns),
        "note": "These are the raw input columns the model pipeline was trained on."
    }
    with open("models/metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("Saved schema to models/metadata.json")

if __name__ == "__main__":
    main()
