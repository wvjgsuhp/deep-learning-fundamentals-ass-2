import tensorflow as tf

from cnns.custom_types import X, Y
from cnns.metrics import Metrics
from cnns.models import VGG16, VGG19, ResNet50
from cnns.safe_io import read_cifar10_test, read_cifar10_train
from cnns.utils import parse_config, train_test_split

if __name__ == "__main__":
    config = parse_config("./config.yaml")
    data: DataSet = {"train": {}, "validation": {}, "test": {}}

    y: Y = {}
    x_train, y_train = read_cifar10_train()
    x_train, y["train"], x_validate, y["validation"] = train_test_split(x_train, y_train, test_ratio=0.2)
    y_train = tf.keras.utils.to_categorical(y["train"], 10)
    y_validate = tf.keras.utils.to_categorical(y["validation"], 10)

    x_test, y["test"] = read_cifar10_test()

    for Model in [VGG16, VGG19, ResNet50]:
        x: X = {
            "train": Model.transform(x_train),
            "validation": Model.transform(x_validate),
            "test": Model.transform(x_test),
        }
        for lr in config["learning_rates"]:
            for layers in config["architectures"]:
                model = Model(layers, learning_rate=lr)
                model.fit(
                    x["train"],
                    y_train,
                    epochs=10000,
                    batch_size=512,
                    validation_data=(x["validation"], y_validate),
                )
                y_pred = model.predict(x_test)
                prediction_class = Model.get_prediction_classes(y_pred)
                print(f"Accuracy: {Model.get_accuracy(y_test, prediction_class)}")
