import tensorflow as tf
from tensorflow import keras


""""
subclassing torch model tanimlamaya cok benziyor her seyi sen yapiyosun ve
static graph olusturamadi icin model her seyi kendine yapmak zorunasin model save weight gibi vs

bu tarz bir yazimda call icerisinde torch gibi daha low level islemler yapilabilri

if conditions
looplar vs vs

bunlari functional API de yapamzsin

FAKAT

model architecture i hidden oldugu icin summary gibi method lar calismaz ev inputlarin boyutlarina, type larina dikkat
etmek gerekir
"""


class SubclassAPI(keras.Model):

    def __init__(self, units=10, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = keras.layers.Dense(units, activation=activation)
        self.hidden1 = keras.layers.Dense(units, activation=activation)
        self.main_output = keras.layers.Dense(1)
        self.aux_output = keras.layers.Dense(1)

    def call(self, inputs):
        input_A, input_B = inputs

        hidden1 = self.hidden1(input_B)
        hidden2 = self.hidden2(hidden1)
        concat = keras.layers.concatenate([input_A, hidden2])
        main_output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        return main_output, aux_output


model = SubclassAPI()




