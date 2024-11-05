import tensorflow as tf

from cnns.custom_types import X, Y
from cnns.metrics import Metrics
from cnns.models import VGG16, VGG19, ResNet50
from cnns.safe_io import read_cifar10_test, read_cifar10_train
from cnns.utils import parse_config, train_test_split

if __name__ == "__main__":
    config = parse_config("./config.yaml")

    y: Y = {}
    x_train, y_train = read_cifar10_train()
    x_train, y["train"], x_validate, y["validation"] = train_test_split(
        x_train, y_train, test_ratio=0.2, random_seed=config["random_seed"]
    )
    y_train = tf.keras.utils.to_categorical(y["train"], 10)
    y_validate = tf.keras.utils.to_categorical(y["validation"], 10)

    x_test, y["test"] = read_cifar10_test()
    metrics = Metrics("./metrics.csv", y.keys())

    batch_size = 512

    for Model in [VGG16, VGG19, ResNet50]:
        x: X = {
            "train": Model.transform(x_train),
            "validation": Model.transform(x_validate),
            "test": Model.transform(x_test),
        }
        for lr in config["learning_rates"]:
            for layers in config["architectures"]:
                model = Model(layers, learning_rate=lr, random_seed=config["random_seed"])
                model.fit(
                    x["train"],
                    y_train,
                    epochs=10000,
                    batch_size=batch_size,
                    validation_data=(x["validation"], y_validate),
                )

                predictions = {
                    "train": Model.get_prediction_classes(model.predict(x["train"])),
                    "validation": Model.get_prediction_classes(model.predict(x["validation"])),
                    "test": Model.get_prediction_classes(model.predict(x["test"])),
                }

                parameters = {
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "base_model": Model.__name__,
                    "architecture": layers,
                }
                accuracies = Metrics.get_accuracies(y, predictions)

                metrics.write(parameters, accuracies)
