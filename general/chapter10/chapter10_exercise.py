import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.metrics import AUC, Precision
# MNIST dataset

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)


def MLP(input_shape, hidden_layers, dnn_units, num_class):
    input_ = keras.layers.Input(input_shape)
    flatten = keras.layers.Flatten()(input_)
    for i in range(hidden_layers):
        if i == 0:
            dnn_in = keras.layers.Dense(dnn_units, activation="relu")(flatten)
            continue
        dnn_in = keras.layers.Dense(dnn_units, activation="relu")(dnn_in)
    output = keras.layers.Dense(num_class, activation="softmax")(dnn_in)
    return keras.Model(inputs=input_, outputs=output)


model = MLP(input_shape=[28, 28], hidden_layers=4, dnn_units=300, num_class=10)
model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=0.0001),
              loss=keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=[AUC(), Precision()])

history = model.fit(X_train, y_train, validation_split=0.1,
          epochs=4, verbose=1,
          callbacks=[keras.callbacks.EarlyStopping(patience=5, monitor="val_loss")])

model.evaluate(X_test, y_test)
print(history.epoch)
import matplotlib.pyplot as plt

fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
axs.plot(history.epoch, history.history["loss"])
axs.plot(history.epoch, history.history["auc"])
axs.plot(history.epoch, history.history["precision"])
axs.plot(history.epoch, history.history["val_loss"])
axs.plot(history.epoch, history.history["val_auc"])
axs.plot(history.epoch, history.history["val_precision"])
axs.legends(["loss", "auc", "precision", "val_loss", "val_auc", "val_precision"])

plt.show()
