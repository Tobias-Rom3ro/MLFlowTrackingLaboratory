import os
from sklearn.linear_model import LogisticRegression
from typing import Dict
import numpy as np

from ..config import RANDOM_STATE, LogRegConfig
from ..metrics import (
    compute_metrics, get_classification_report_dict,
    save_confusion_matrix_png, save_json
)
from ..tracking import run, log_params, log_metrics, log_artifact


def train_and_log_logreg(
        X_train, y_train, X_test, y_test,
        C: float, solver: str, max_iter: int = 1000
) -> Dict[str, any]:
    run_name = f"logreg_C{C}_solver{solver}"

    tags = {
        "dataset": "QSAR_Biodeg",
        "framework": "sklearn",
        "task": "binary_classification"
    }

    with run(run_name, tags=tags):
        model = LogisticRegression(
            C=C, solver=solver, max_iter=max_iter, random_state=RANDOM_STATE
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = compute_metrics(y_test, y_pred)

        params = {
            "C": C,
            "solver": solver,
            "max_iter": max_iter,
            "random_state": RANDOM_STATE
        }
        log_params(params)
        log_metrics(metrics)

        cm_path = "confusion_matrix_logreg.png"
        save_confusion_matrix_png(y_test, y_pred, cm_path)
        log_artifact(cm_path)
        os.remove(cm_path)

        report_dict = get_classification_report_dict(y_test, y_pred)
        report_path = "classification_report_logreg.json"
        save_json(report_dict, report_path)
        log_artifact(report_path)
        os.remove(report_path)

    return {**params, **metrics}


def grid_search_logreg(X_train, y_train, X_test, y_test, config: LogRegConfig) -> Dict[str, any]:
    best_result = {"f1_score": 0}

    for C in config.Cs:
        for solver in config.solvers:
            result = train_and_log_logreg(
                X_train, y_train, X_test, y_test,
                C=C, solver=solver, max_iter=config.max_iter
            )
            if result["f1_score"] > best_result["f1_score"]:
                best_result = result

    return best_result