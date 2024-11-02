import tensorflow as tf
from tensorflow.keras.applications import vgg16

from .base import BaseModel


class VGG16(BaseModel):
    def __init__(self) -> None:
        super(VGG16, self).__init__()

        base_model = vgg16.VGG16(input_shape=(32, 32, 3), include_top=False, weights="imagenet")
        base_model.trainable = False

        x = tf.keras.layers.Flatten()(base_model.output)
        x = self._dense_relu(512)(x)
        x = self._dense_relu(256)(x)
        y = self._dense(10, activation="softmax")(x)

        self._model = tf.keras.Model(inputs=x, outputs=y)
