import os
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

from .config import (
    RANDOM_STATE, EXPERIMENT_NAME, LogRegConfig, KerasConfig
)
from .data import (
    load_qsar_biodeg, prepare_data, split_and_scale, get_class_distribution
)
from .models import grid_search_logreg, train_and_log_keras
from .llm import generate_insights
from .tracking import set_experiment, run, log_artifact, enable_global_autolog, init_tracking


def main(full: bool = False):
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    print("[INFO] Cargando dataset QSAR Biodegradation")
    df = load_qsar_biodeg()
    X, y = prepare_data(df)

    print(f"[INFO] Dataset: {X.shape[0]} muestras, {X.shape[1]} features")
    print(f"[INFO] Distribución de clases: {get_class_distribution(y)}")

    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
    print(f"[INFO] Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    init_tracking()
    set_experiment(EXPERIMENT_NAME)
    print(f"[INFO] Experimento MLflow: {EXPERIMENT_NAME}")

    print("[INFO] Iniciando Regresión Logística")
    logreg_config = LogRegConfig()
    best_logreg = grid_search_logreg(X_train, y_train, X_test, y_test, logreg_config)
    print(
        f"[INFO] Mejor Logística: C={best_logreg['C']}, solver={best_logreg['solver']}, F1={best_logreg['f1_score']:.4f}")

    print("[INFO] Iniciando Red Neuronal")
    keras_config = KerasConfig()
    input_dim = X_train.shape[1]
    keras_metrics = train_and_log_keras(X_train, y_train, X_test, y_test, input_dim, keras_config)
    print(f"[INFO] Red Neuronal: F1={keras_metrics['f1_score']:.4f}, Accuracy={keras_metrics['accuracy']:.4f}")

    print("[INFO] Interpretación con Gemini")
    prompt_text = f"""
Analiza los siguientes resultados de clasificación binaria sobre el dataset QSAR Biodegradation:

Mejor modelo Logístico:
- F1-score: {best_logreg['f1_score']:.4f}
- Accuracy: {best_logreg['accuracy']:.4f}
- Recall: {best_logreg['recall']:.4f}
- Hiperparámetros: C={best_logreg['C']}, solver={best_logreg['solver']}

Modelo Red Neuronal:
- F1-score: {keras_metrics['f1_score']:.4f}
- Accuracy: {keras_metrics['accuracy']:.4f}
- Recall: {keras_metrics['recall']:.4f}

Responde brevemente:
1. ¿Qué significa un F1-score de {best_logreg['f1_score']:.4f} considerando el balance de clases en este dataset?
2. ¿Por qué la red neuronal podría tener un recall de {keras_metrics['recall']:.4f} comparado con {best_logreg['recall']:.4f} de regresión logística?
3. Sugiere dos mejoras concretas para optimizar estos modelos en este dataset.
"""

    gemini_text = generate_insights(prompt_text)
    print("[INFO] Gemini insights generados")

    gemini_path = "gemini_insights.txt"
    with open(gemini_path, 'w') as f:
        f.write(gemini_text)

    with run("gemini_interpretation", tags={"analysis_type": "gemini_insights"}):
        log_artifact(gemini_path)

    os.remove(gemini_path)

    if full:
        print("[INFO] Ejecutando runs anidados")
        enable_global_autolog()

        with run("nested_experiments_root"):
            with run("nested_logreg_C5", nested=True):
                model_nested1 = LogisticRegression(
                    C=5.0, solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE
                )
                model_nested1.fit(X_train, y_train)
                y_pred_nested1 = model_nested1.predict(X_test)
                f1_nested1 = f1_score(y_test, y_pred_nested1)
                print(f"[INFO] Nested LogReg: F1={f1_nested1:.4f}")

            with run("nested_keras_e30_bs16", nested=True):
                model_nested2 = Sequential([
                    Dense(32, activation='relu', input_shape=(input_dim,)),
                    Dense(16, activation='relu'),
                    Dropout(0.2),
                    Dense(1, activation='sigmoid')
                ])
                model_nested2.compile(
                    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']
                )
                model_nested2.fit(
                    X_train, y_train,
                    epochs=30,
                    batch_size=16,
                    validation_split=0.2,
                    verbose=0
                )
                y_pred_prob_nested2 = model_nested2.predict(X_test, verbose=0)
                y_pred_nested2 = (y_pred_prob_nested2 > 0.5).astype(int).flatten()
                f1_nested2 = f1_score(y_test, y_pred_nested2)
                print(f"[INFO] Nested Keras: F1={f1_nested2:.4f}")

        print("[INFO] Runs anidados completados")

    print("[INFO] Proceso completado exitosamente")