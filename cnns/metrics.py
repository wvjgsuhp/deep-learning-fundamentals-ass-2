import os
from datetime import datetime
from typing import Iterable

import numpy as np

from .custom_types import Accuracies, Layers, Y


class Metrics:
    def __init__(self, metric_path: str, set_types: Iterable[str]) -> None:
        self.metric_path = metric_path
        self.set_types = set_types
        self.parameters = ["batch_size", "learning_rate", "base_model", "architecture"]

        self.write_header()

    def write_header(self) -> None:
        if not os.path.isfile(self.metric_path):
            metrics = ["accuracy"]

            metric_cols = "|".join(
                [f"{set_type}_{metric}" for set_type in self.set_types for metric in metrics]
            )
            param_cols = "|".join(self.parameters)

            with open(self.metric_path, "a") as f:
                f.write(f"saved_at|{metric_cols}|{param_cols}\n")

    def write(self, parameters: dict[str, object], accuracies: Accuracies) -> None:
        saved_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.metric_path, "a") as f:
            serialized_metrics = "|".join(map(str, (accuracies[s] for s in self.set_types)))
            serialized_params = "|".join(map(str, (parameters[p] for p in self.parameters)))

            f.write(f"{saved_at}|{serialized_metrics}|{serialized_params}\n")

    @staticmethod
    def get_accuracies(truth: Y, predictions: Y) -> Accuracies:
        return {set_type: np.mean(t == predictions[set_type]) for set_type, t in truth.items()}
