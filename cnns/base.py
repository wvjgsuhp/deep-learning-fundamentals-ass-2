from typing import Any

import numpy as np
import tensorflow as tf

from .custom_types import MI, Layers, NPFloatMatrix, NPInt, copy_args

tf_initializers = tf.keras.initializers
tf_layers = tf.keras.layers


class BaseTransferClassifier:
    def __init__(
        self,
        layers: Layers,
        learning_rate: float = 0.0001,
        loss: str = "categorical_crossentropy",
        random_seed: int | None = None,
    ) -> None:
        tf.random.set_seed(random_seed)
        tf.keras.utils.set_random_seed(random_seed)

        self.is_model_compiled = False
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss = loss

        self._model = self.init_model(layers)

    def init_model(self, layers: Layers) -> tf.keras.Model:
        base_model = self._get_base_model()
        base_model.trainable = False

        x = tf.keras.layers.Flatten()(base_model.output)
        x = self._get_learning_layers(layers, x)
        y = TFLayer.dense(10, activation="softmax")(x)

        return tf.keras.Model(inputs=base_model.input, outputs=y)

    @staticmethod
    def transform(x: MI) -> MI:
        raise NotImplementedError()

    @property
    def model(self) -> tf.keras.Model:
        if not self.is_model_compiled:
            self._model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=self.loss)
            self.is_model_compiled = True

        return self._model

    @copy_args(tf.keras.Model.fit)
    def fit(self, *args, **kwargs: Any) -> None:
        early_stopping = tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
        kwargs["callbacks"] = [early_stopping]

        self.model.fit(*args, **kwargs)

    @copy_args(tf.keras.Model.predict)
    def predict(self, *args, **kwargs: Any) -> NPFloatMatrix:
        return self.model.predict(*args, **kwargs)

    @staticmethod
    def get_prediction_classes(y: NPFloatMatrix) -> NPInt:
        return np.argmax(y, axis=1)

    @staticmethod
    def get_accuracy(truth: NPInt, prediction: NPInt) -> float:
        return np.mean(truth == prediction)

    def _get_base_model(self) -> tf.keras.Model:
        raise NotImplementedError()

    def _get_learning_layers(self, layers: Layers, x: tf.Tensor) -> tf.Tensor:
        return TFLayer.get_tensor_from_config(layers, x)


class TFLayer:
    @staticmethod
    def dense(units: int, **kwargs: Any) -> tf_layers.Dense:
        return tf_layers.Dense(units, kernel_initializer=tf_initializers.HeNormal(), **kwargs)

    @staticmethod
    def dense_relu(units: int, **kwargs: Any) -> tf_layers.Dense:
        return TFLayer.dense(units, activation="relu", **kwargs)

    @staticmethod
    def mlp(x: tf.Tensor, units: int, n: int, **_: Any) -> tf.Tensor:
        if n == 0:
            return x

        layer = TFLayer.dense_relu(units)(x)
        return TFLayer.mlp(layer, units, n - 1)

    @staticmethod
    def get_tensor_from_config(layers: Layers, input_: tf.Tensor) -> tf.Tensor:
        tensor = input_
        for layer in layers:
            kwargs = layer.copy()
            layer_name = kwargs.pop("layer", None)

            match layer_name:
                case "dense":
                    tensor = TFLayer.dense(**kwargs)(tensor)
                case "dense_relu":
                    tensor = TFLayer.dense_relu(**kwargs)(tensor)
                case "mlp":
                    tensor = TFLayer.mlp(tensor, **kwargs)
                case _:
                    raise NotImplementedError(f"Unknown layer {layer_name}")

        return tensor
