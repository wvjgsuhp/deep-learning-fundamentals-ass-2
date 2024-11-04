import os


class Metrics:
    def __init__(self, metric_path: str, data_sets: list[str]) -> None:
        self.metric_path = metric_path
        self.data_sets = data_sets

        self.write_header()

    def write_header(self) -> None:
        if not os.path.isfile(self.metric_path):
            metrics = ["accuracy"]
            metric_cols = "|".join(
                [f"{set_type}_{metric}" for set_type in self.data_sets for metric in metrics]
            )
            with open(self.metric_path, "a") as f:
                f.write(f"saved_at|{metric_cols}|param_id\n")
