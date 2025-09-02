import numpy as np
from joblib import dump
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor

from data import load_california
from features import split_columns, build_preprocessor

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def main():
    # 1) load data (log target)
    X, y = load_california(as_log_target=True)

    # 2) preprocessor
    num_cols, cat_cols = split_columns(X)
    pre = build_preprocessor(num_cols, cat_cols)

    # 3) model (LightGBM)
    model = LGBMRegressor(
        n_estimators=800,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    pipe = Pipeline([("pre", pre), ("model", model)])

    # 4) cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_rmse = -cross_val_score(pipe, X, y, scoring="neg_root_mean_squared_error", cv=cv)
    print(f"CV RMSE (log target): mean={cv_rmse.mean():.4f}, std={cv_rmse.std():.4f}")

    # 5) final fit & save
    pipe.fit(X, y)
    dump(pipe, "models/house_price_pipeline.joblib")
    print("Saved model to models/house_price_pipeline.joblib")

if __name__ == "__main__":
    main()
