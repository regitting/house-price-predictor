from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

def split_columns(X: pd.DataFrame):
    num_cols = selector(dtype_include=np.number)(X)
    cat_cols = selector(dtype_exclude=np.number)(X)
    return num_cols, cat_cols

def build_preprocessor(num_cols, cat_cols):
    num = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("num", num, num_cols),
        ("cat", cat, cat_cols)
    ])