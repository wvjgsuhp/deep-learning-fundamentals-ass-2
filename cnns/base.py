from typing import Any

import tensorflow as tf

from .custom_types import NPFloatMatrix, copy_args

tf_initializers = tf.keras.initializers
tf_layers = tf.keras.layers

MarkerModel = tf.keras.Model


class BaseModel:
    def __init__(self) -> None:
        self.is_model_compiled = False

    @property
    def model(self) -> MarkerModel:
        if not self.is_model_compiled:
            self._model.compile(optimizer=self.optimizer, loss=self.loss)
            self.is_model_compiled = True

        return self._model

    @copy_args(tf.keras.Model.fit)
    def fit(self, *args, **kwargs: Any) -> None:
        self.model.fit(*args, **kwargs)

    @copy_args(tf.keras.Model.predict)
    def predict(self, *args, **kwargs: Any) -> NPFloatMatrix:
        return self.model.predict(*args, **kwargs)

    def _dense(self, units: int, **kwargs: Any) -> tf_layers.Dense:
        return tf_layers.Dense(units, kernel_initializer=tf_initializers.HeNormal(), **kwargs)

    def _dense_relu(self, units: int, **kwargs: Any) -> tf_layers.Dense:
        return self._dense(units, activation="relu", **kwargs)

    def _mlp(self, x: tf.Tensor, units: int, n: int) -> tf.Tensor:
        if n == 0:
            return x

        layer = self._dense_relu(units)(x)
        return self._mlp(layer, units, n - 1)
