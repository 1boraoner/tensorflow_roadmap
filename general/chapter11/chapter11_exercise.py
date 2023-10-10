import os 
os.environ['TF_CPP_LEVEL']='3'

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


def DNN(input_shape, use_bn):
    input_ = keras.layers.Input(input_shape)
    flatten = keras.layers.Flatten()(input_)
    # if use_bn:
    #     flatten = keras.layers.BatchNormalization()(flatten)
    for i in range(5):
        if i == 0 :
            dnn_in = keras.layers.Dense(units = 100, activation = "elu", kernel_initializer="he_normal")(flatten)
            if use_bn:
                dnn_in = keras.layers.BatchNormalization()(dnn_in)
            continue
        dnn_in = keras.layers.Dense(units=100, activation = "elu", kernel_initializer="he_normal")(dnn_in)
        if use_bn:
            dnn_in = keras.layers.BatchNormalization()(dnn_in)
    
    output = keras.layers.Dense(5, activation = "softmax")(dnn_in)
    return keras.Model(inputs=input_, outputs = output)


#dataset ops

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
print(type(X_train))
mask_train = y_train < 5
mask_test = y_test < 5

X_train_04, y_train_04 = X_train[mask_train], keras.utils.to_categorical(y_train[mask_train])
X_test_04, y_test_04 = X_test[mask_test], keras.utils.to_categorical(y_test[mask_test])


#regular training only with classes 0 and 4 
model = DNN(input_shape = X_train_04.shape[1:], use_bn=False)
model.compile(optimizer = keras.optimizers.legacy.Adam(learning_rate = 0.001),
              loss = keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics = [keras.metrics.Precision(name="precision_dnn"), 
                         keras.metrics.AUC(name="auc_dnn")])

history = model.fit(X_train_04, y_train_04, validation_split = 0.15, epochs=100, verbose=1, 
                    callbacks=[keras.callbacks.EarlyStopping(monitor='val_precision_dnn', 
                                                            patience=15, 
                                                            mode= "max",
                                                            verbose=1),
                                keras.callbacks.ModelCheckpoint(filepath = "../../model_checkpoints/chapter11/dnn", 
                                                                monitor='val_precision_dnn', 
                                                                verbose=1, 
                                                                mode= "max",
                                                                save_best_only=True)])


model_bn = DNN(input_shape = X_train_04.shape[1:], use_bn=True)
model_bn.compile(optimizer = keras.optimizers.legacy.Adam(learning_rate = 0.001),
              loss = keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics = [keras.metrics.Precision(name="precision_bn"), 
                         keras.metrics.AUC(name = "auc_bn")])

history_bn = model_bn.fit(  X_train_04, y_train_04, validation_split = 0.15, epochs=100, verbose=1, 
                            callbacks=[keras.callbacks.EarlyStopping(monitor='val_precision_bn', 
                                                                    patience=15, 
                                                                    mode= "max",
                                                                    verbose=1),
                                        keras.callbacks.ModelCheckpoint(filepath = "../../model_checkpoints/chapter11/dnn_bn", 
                                                                        monitor='val_precision_bn', 
                                                                        verbose=1, 
                                                                        mode= "max",
                                                                        save_best_only=True)])

fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize=(10,10))
axs[0].plot(history.epoch, history.history["loss"], label='dnn_loss')
axs[0].plot(history.epoch, history.history["val_loss"], label='dnn_val_loss')
axs[0].plot(history.epoch, history.history["precision_dnn"], label='dnn_precision')
axs[0].plot(history.epoch, history.history["val_precision_dnn"], label='dnn_val_precision')
axs[0].set_ylim([0, 2])
axs[0].legend()

axs[1].plot(history_bn.epoch, history_bn.history["loss"], label='dnn_bn_loss')
axs[1].plot(history_bn.epoch, history_bn.history["val_loss"], label='dnn_bn_val_loss')
axs[1].plot(history_bn.epoch, history_bn.history["precision_bn"], label='dnn_bn_precision')
axs[1].plot(history_bn.epoch, history_bn.history["val_precision_bn"], label='dnn_bn_val_precision')
axs[1].set_ylim([0, 1.2])
axs[1].legend()

plt.show()
plt.savefig('dnn_bn_vs.png')