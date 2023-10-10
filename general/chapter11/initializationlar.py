import tensorflow as tf
from tensorflow import keras
import numpy as np
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[10]),
    keras.layers.Dense(units=50, activation='selu', kernel_initializer= "he_normal", name='dnn_1'),
    keras.layers.Dense(units=1, activation='sigmoid', kernel_initializer="he_normal", name="out_layer")
])

for layer in model.layers:
    try:
        weights = layer.get_weights()[0]
        print(layer.get_config()["name"], weights)
        print(weights.shape)
        print("mean ", np.mean(weights))
        print("std ", round(np.std(weights)**2, 2))

        print("fan_in ", weights.shape[0])
        print("std_from_fan_in ", 1/(weights.shape[0]))

    except:
        pass

