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
    possible_targets = ["Class", "class", "target"]
    target_col = None

    for col in possible_targets:
        if col in df.columns:
            target_col = col
            break

    if target_col is None:
        raise ValueError(f"No se encontró columna objetivo. Columnas disponibles: {df.columns.tolist()}")

    X = df.drop(columns=[target_col])
    y_raw = df[target_col]

    unique_values = y_raw.unique()

    mapping_options = [
        {'NRB': 0, 'RB': 1},
        {'1': 0, '2': 1},
        {1: 0, 2: 1},
        {'0': 0, '1': 1},
        {0: 0, 1: 1}
    ]

    y = None
    for mapping in mapping_options:
        y_temp = y_raw.map(mapping)
        if not y_temp.isna().any():
            y = y_temp
            break

    if y is None or y.isna().any():
        raise ValueError(f"No se pudo mapear la columna {target_col}. Valores encontrados: {unique_values}")

    return X, y


def split_and_scale(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    y_array = y.astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_array, test_size=test_size, stratify=y_array, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def get_class_distribution(y: pd.Series) -> dict:
    return y.value_counts().to_dict()