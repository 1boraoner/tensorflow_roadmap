import tensorflow as tf
from tensorflow import keras

#use model A's all layers excepy output layer


# burda A ile B nin ayni weightler oldugu icin B de yaptigin degisiklikler A yi da etkiler!!!
model_A = keras.model.load_model("my_model_A.h5")
model_B_on_A = keras.models.Sequential(model_A.layers[:-1])
model_B_on_A.add(keras.layers.Dense(1, activation='sigmoid'))

# bunu onlemek icin clone lamak lazim

model_A_clone = keras.models.close_model(model_A)
model_A_clone.set_weigths(model_A.get_weights())

#freeze the other layers

for layer in model_B_on_A.layers[:-1]:
    layer.trainable = False

# freeze or unfreeze sonrasi compile alinmalidir
model_B_on_A.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])

"""
Next, we can train the model for a few epochs, then unfreeze the reused layers (which
requires compiling the model again) and continue training to fine-tune the reused
layers for task B. After unfreezing the reused layers, it is usually a good idea to reduce
the learning rate, once again to avoid damaging the reused weights:
"""

