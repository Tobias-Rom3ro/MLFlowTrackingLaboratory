import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple

from .config import RANDOM_STATE


def load_qsar_biodeg() -> pd.DataFrame:
    data = fetch_openml(data_id=1494, as_frame=True, parser='auto')
    return data.frame


def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    target_col = "Class" if "Class" in df.columns else "class"
    X = df.drop(columns=[target_col])
    y = df[target_col].map({'NRB': 0, 'RB': 1})
    return X, y


def split_and_scale(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def get_class_distribution(y: pd.Series) -> dict:
    return y.value_counts().to_dict()