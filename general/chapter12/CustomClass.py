import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

class MyCNN(keras.Model):
    def __init__(self, num_classes,  **kwargs):
        super(MyCNN, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.flatten = keras.layers.Flatten()
        self.dnn1 = keras.layers.Dense(300, 100, activation="relu",
                                      kernal_regularizer=1.e-5, kernel_initializer="he_normal")
        self.out = keras.layers.Dense(100, self.num_classes, activation="softmax",
                                       kernal_regularizer=1.e-5, kernel_initializer="he_normal")

    def call(self, inputs):
        Z = self.flatten(inputs)
        Z = self.dnn1(Z)
        Z = self.out(Z)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "num_classes":self.num_classes)


