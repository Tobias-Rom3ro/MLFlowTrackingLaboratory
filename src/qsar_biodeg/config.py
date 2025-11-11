import os
from dataclasses import dataclass
from typing import Tuple
import dotenv

RANDOM_STATE = 42
EXPERIMENT_NAME = "qsar_biodeg_experiments"
GENAI_MODEL = "gemini-2.5-flash"
dotenv.load_dotenv()


@dataclass
class LogRegConfig:
    Cs: Tuple[float, ...] = (0.1, 1.0, 10.0)
    solvers: Tuple[str, ...] = ("liblinear", "lbfgs")
    max_iter: int = 1000


@dataclass
class KerasConfig:
    hidden1: int = 32
    hidden2: int = 16
    dropout: float = 0.2
    epochs: int = 50
    batch_size: int = 32
    validation_split: float = 0.2


def get_genai_api_key() -> str:
    return os.getenv("GOOGLE_API_KEY")