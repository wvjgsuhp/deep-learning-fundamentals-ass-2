#!./env/bin/python
import os
import re
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def initialize_dir(dir_name: str) -> None:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def get_violin_plot(df: pd.DataFrame, file_path: str) -> None:
    fig, ax = plt.subplots(figsize=(4, 2.5))

    models = ["VGG16", "VGG19", "ResNet50"]

    accuracies = []
    for model in models:
        df_sub = df[df.base_model.eq(model)]
        accuracies.append(df_sub.validation_accuracy)

    ax.violinplot(
        accuracies,
        showmedians=True,
    )

    labels = models
    ax.set_xticks(range(1, len(labels) + 1), labels=labels)

    plt.rcParams.update({"font.size": 21})
    plt.xlabel("Base model")
    plt.ylabel("Accuracy")
    plt.savefig(file_path, bbox_inches="tight")


def get_latex_table(df: pd.DataFrame, latex_table_file: str, **to_latex_kwargs: Any) -> None:
    # map the architectures to symbols
    layer_mapping = {
        "[]": r"\\architecturea{-0.1cm}",
        "[{'layer': 'dense_relu', 'units': 512}, {'layer': 'dense_relu', 'units': 256}]": r"\\architectureb{-0.1cm}",
        "[{'layer': 'dense_relu', 'units': 1024}, {'layer': 'dense_relu', 'units': 512}]": r"\\architecturec{-0.1cm}",
        "[{'layer': 'dense_relu', 'units': 2048}, {'layer': 'dense_relu', 'units': 1024}]": r"\\architectured{-0.1cm}",
        "[{'layer': 'dense_relu', 'units': 4096}, {'layer': 'dense_relu', 'units': 2048}]": r"\\architecturee{-0.1cm}",
    }

    accuracy_formatter = {
        "learning_rate": float,
        "train_accuracy": "{:.3f}".format,
        "validation_accuracy": "{:.3f}".format,
        "test_accuracy": "{:.3f}".format,
    }

    # create raw latex
    latex_table_raw = (
        df.to_latex(
            index=False,
            formatters=accuracy_formatter,
            **to_latex_kwargs,
        )
        .replace("& Architecture ", "")
        .replace("table", "table*")
    )

    latex_table = []
    row_count = 0
    # modify the style of the table
    for line in latex_table_raw.split("\n"):
        if line.startswith(r"\begin{table"):
            latex_table += [line + "[t]"]
        elif line.startswith(r"\begin{tabul"):
            latex_table += [r"\begin{tabularx}{\textwidth}{LLllLl}"]
        elif line.startswith("base"):
            latex_table += [r"  \rowcolor{lightgray}"]
            latex_table += [
                r"  \bf Base  & \bf Learning  & \bf Architecture  & \multicolumn{3}{c}{\bf Accuracy} \\"
            ]
            latex_table += [r"  \rowcolor{lightgray}"]
            latex_table += [
                r"  \bf model & \bf rate      &                   & \bf Train & \bf Validat-ion & \bf Test \\"
            ]
            latex_table += [r"  \bhline"]
        elif line.endswith("rule"):
            continue
        elif line.startswith(("ResNet50", "VGG")):
            if row_count % 2 == 1:
                latex_table += [r"  \evenrow"]

            layer = re.match(r".*(\[.*\])", line).group(1)
            latex_table += ["  " + re.sub(r"\[.*\]", layer_mapping[layer], line)]

            row_count += 1

        elif line.startswith(r"\end{tabu"):
            latex_table[-1] = latex_table[-1][:-3]
            latex_table += [r"\end{tabularx}"]
        else:
            latex_table += [line]

    latex_table_str = "\n".join(
        (f"  {line}" if i > 0 and i < len(latex_table) - 2 else line for i, line in enumerate(latex_table))
    )

    with open(latex_table_file, "w") as f:
        f.write(latex_table_str)


if __name__ == "__main__":
    # create a directory for assets
    asset_dir = "./assets"
    initialize_dir(asset_dir)

    df_metric = pd.read_csv("./metrics.csv", sep="|")

    violin_plot_file = os.path.join(asset_dir, "violin_plot.pdf")
    get_violin_plot(df_metric, violin_plot_file)

    # write to latex table
    use_cols = [
        "base_model",
        "learning_rate",
        "architecture",
        "train_accuracy",
        "validation_accuracy",
        "test_accuracy",
    ]

    latex_table_file = "./latex/best-performance.tex"
    df_best = (
        df_metric.groupby(["base_model"])
        .validation_accuracy.max()
        .reset_index()
        .merge(df_metric, on=["base_model", "validation_accuracy"])[use_cols]
    )
    get_latex_table(
        df_best,
        latex_table_file,
        caption="The best performance of each model on the validation set",
        label="tab:model-performance",
    )

    latex_table_file = "./latex/models-performance.tex"
    df_example = df_metric[df_metric.learning_rate.eq(0.001)][use_cols]
    get_latex_table(
        df_example,
        latex_table_file,
        caption="The performance of models on the validation set with a learning rate of 0.001",
        label="tab:models-performance",
    )
