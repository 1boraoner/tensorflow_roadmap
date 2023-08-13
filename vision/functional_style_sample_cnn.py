import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Accuracy, AUC


# Functional CNN Model

input_layer = Input(shape=(220,220, 3))

conv1 = Conv2D(filters=32, kernel_size = (5,5), strides=(1,1), padding='valid', activation='relu', use_bias='True')(input_layer)
pool1 = MaxPooling2D(pool_size = (2,2), strides=None, padding='valid')(conv1)
conv2 = Conv2D(filters=64, kernel_size = (5,5), strides=(1,1), padding='valid', activation='relu', use_bias='True')(pool1)
pool2 = MaxPooling2D(pool_size = (2,2), strides=None, padding='valid')(conv2)
conv3 = Conv2D(filters=128, kernel_size = (5,5), strides=(1,1), padding='valid', activation='relu', use_bias='True')(pool2)
pool3 = MaxPooling2D(pool_size = (2,2), strides=None, padding='valid')(conv3)
flattened = Flatten()(pool3)
dense_1 = Dense(units=128, activation='relu')(flattened) 
output = Dense(units=2, activation='sigmoid')(dense_1) 

func_model = Model(inputs= input_layer, outputs=output)

func_model.compile(  optimizer=SGD(learning_rate = 0.0001,),
                    loss = BinaryCrossentropy(from_logits=False), 
                    metrics=[AUC(),
                             Accuracy()]
                )

func_model.summary()