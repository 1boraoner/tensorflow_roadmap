import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import  Input, Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import AUC, Accuracy
import numpy as np

#Sequential Model Definition

seq_model = tf.keras.Sequential(layers = [
    Conv2D(filters=32, kernel_size = (5,5), strides=(1,1), padding='valid', activation='relu', use_bias='True'),
    MaxPooling2D(pool_size = (2,2), strides=None, padding='valid'),
    Conv2D(filters=64, kernel_size = (5,5), strides=(1,1), padding='valid', activation='relu', use_bias='True'),
    MaxPooling2D(pool_size = (2,2), strides=None, padding='valid'),
    Conv2D(filters=128, kernel_size = (5,5), strides=(1,1), padding='valid', activation='relu', use_bias='True'),
    MaxPooling2D(pool_size = (2,2), strides=None, padding='valid'),
    Flatten(),
    Dense(units=128, activation = 'relu', use_bias='True'),
    Dense(units=2, activation='sigmoid', use_bias=False)
    ],
    name = "Sequential CNN Model"
)

seq_model.build(input_shape=(None, 224,224,3))

seq_model.compile(  optimizer=SGD(learning_rate = 0.0001,),
                    loss = BinaryCrossentropy(from_logits=False), 
                    metrics=[AUC(),
                             Accuracy()]
                )

seq_model.summary()
