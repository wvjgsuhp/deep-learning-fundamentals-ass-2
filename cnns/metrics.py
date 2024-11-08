import os
from datetime import datetime
from typing import Iterable

import numpy as np

from .custom_types import Accuracies, Y


class Metrics:
    def __init__(self, metric_file: str, set_types: Iterable[str], target_file: str = "y.csv") -> None:
        self.metric_file = metric_file
        self.set_types = set_types
        self.parameters = ["batch_size", "learning_rate", "base_model", "architecture"]
        self.target_file = target_file

        self.clean_target_file(target_file)
        self.write_header()

    def write_header(self) -> None:
        if not os.path.isfile(self.metric_file):
            metrics = ["accuracy"]

            metric_cols = "|".join(
                [f"{set_type}_{metric}" for set_type in self.set_types for metric in metrics]
            )
            param_cols = "|".join(self.parameters)

            with open(self.metric_file, "a") as f:
                f.write(f"saved_at|{metric_cols}|{param_cols}\n")

    def write(self, parameters: dict[str, object], accuracies: Accuracies) -> None:
        saved_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.metric_file, "a") as f:
            serialized_metrics = "|".join(map(str, (accuracies[s] for s in self.set_types)))
            serialized_params = "|".join(map(str, (parameters[p] for p in self.parameters)))

            f.write(f"{saved_at}|{serialized_metrics}|{serialized_params}\n")

    @staticmethod
    def write_target(y: Y, model_name: str, target_file: str) -> None:
        with open(target_file, "a") as f:
            for set_type, target in y.items():
                serialized_y = "|".join(map(str, target))
                f.write(f"{model_name}|{set_type}|{serialized_y}\n")

    @staticmethod
    def get_accuracies(truth: Y, predictions: Y) -> Accuracies:
        return {set_type: np.mean(t == predictions[set_type]) for set_type, t in truth.items()}

    @staticmethod
    def clean_target_file(target_file: str) -> None:
        with open(target_file, "w") as _:
            pass
