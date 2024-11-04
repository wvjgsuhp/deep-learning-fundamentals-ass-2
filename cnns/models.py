import tensorflow as tf
from tensorflow.keras.applications import resnet, vgg16, vgg19

from .base import BaseTransferClassifier
from .custom_types import MI


class VGG16(BaseTransferClassifier):
    def _get_base_model(self) -> tf.keras.Model:
        return vgg16.VGG16(
            input_tensor=tf.keras.layers.Input(shape=(32, 32, 3)), include_top=False, weights="imagenet"
        )

    @staticmethod
    def transform(x: MI) -> MI:
        return vgg16.preprocess_input(x)


class VGG19(BaseTransferClassifier):
    def _get_base_model(self) -> tf.keras.Model:
        return vgg19.VGG19(
            input_tensor=tf.keras.layers.Input(shape=(32, 32, 3)), include_top=False, weights="imagenet"
        )

    @staticmethod
    def transform(x: MI) -> MI:
        return vgg19.preprocess_input(x)


class ResNet50(BaseTransferClassifier):
    def _get_base_model(self) -> tf.keras.Model:
        return resnet.ResNet50(
            input_tensor=tf.keras.layers.Input(shape=(32, 32, 3)), include_top=False, weights="imagenet"
        )

    @staticmethod
    def transform(x: MI) -> MI:
        return resnet.preprocess_input(x)
