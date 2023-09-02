import tensorflow as tf
from tensorflow import keras

#important callbacks
# 1- ModelCheckpoint --> model parametrlerini intervallar icinde saklar
# 2 - EarlyStopping --> early stopping mechanism icin

#writin custome callbacks

class PrintValTrainRatioCallback(keras.callbacks.Callback):

    """
        train_begin
        train_end
        epoch_begin
        epoch_end
        gibi bir cok member function var

    """
    def on_epoch_begin(self, epoch, logs=None):
        pass
    def on_epoch_end(self, epoch, logs):
        # butun metricler log icerisinde tutuluyo
        val_loss = logs['val_loss']
        train_loss = logs["train_loss"]
        print(f"{val_loss / train_loss}")