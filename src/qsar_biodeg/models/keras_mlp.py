import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from typing import Dict

from ..config import RANDOM_STATE, KerasConfig
from ..metrics import compute_metrics, save_confusion_matrix_png
from ..tracking import run, log_metrics, log_artifact, enable_tf_autolog


def build_model(input_dim: int, h1: int = 32, h2: int = 16, dropout: float = 0.2):
    model = Sequential([
        Dense(h1, activation='relu', input_shape=(input_dim,)),
        Dense(h2, activation='relu'),
        Dropout(dropout),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_and_log_keras(
        X_train, y_train, X_test, y_test,
        input_dim: int, config: KerasConfig
) -> Dict[str, float]:
    enable_tf_autolog()

    tags = {
        "dataset": "QSAR_Biodeg",
        "framework": "keras",
        "task": "binary_classification"
    }

    with run("keras_mlp_baseline", tags=tags):
        model = build_model(
            input_dim, config.hidden1, config.hidden2, config.dropout
        )

        early_stop = EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )

        history = model.fit(
            X_train, y_train,
            epochs=config.epochs,
            batch_size=config.batch_size,
            validation_split=config.validation_split,
            callbacks=[early_stop],
            verbose=0
        )

        y_pred_prob = model.predict(X_test, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()

        metrics = compute_metrics(y_test, y_pred)
        log_metrics(metrics)

        cm_path = "confusion_matrix_keras.png"
        save_confusion_matrix_png(y_test, y_pred, cm_path)
        log_artifact(cm_path)
        os.remove(cm_path)

        config_path = "keras_config.txt"
        with open(config_path, 'w') as f:
            f.write(f"input_dim: {input_dim}\n")
            f.write(f"epochs: {config.epochs}\n")
            f.write(f"batch_size: {config.batch_size}\n")
            f.write(f"final_epoch: {len(history.history['loss'])}\n")
        log_artifact(config_path)
        os.remove(config_path)

    return metrics