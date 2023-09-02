import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder

#keras API has dataset utils

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

X_train, X_val = X_train_full[:50000]/255.0 , X_train_full[50000:]/255.0
y_train, y_val = y_train_full[:50000], y_train_full[50000:]

print(X_train.shape)
def CNN_classifier():

    inputs = keras.layers.Input((28,28))
    flatten = keras.layers.Flatten()(inputs)
    dnn1 = keras.layers.Dense(units=300, activation='relu')(flatten)
    dnn2 = keras.layers.Dense(units=100, activation='relu')(dnn1)
    output = keras.layers.Dense(units=10, activation='softmax')(dnn2)
    return keras.Model(inputs=inputs, outputs= output)

""""
    targerlar = [0, 9] arasinda integer olarak belirlenmis
    bu yuzden loss fonksyonunda Sparse_Categorical_Crossentropy kullaniliyor
    
    Eger targetlari One Hot Encodeing den gecirseydik o zaman Categorical_Crossentropy kullanmak gerekirdi
    
    OneHotEncoding Olusturmak cok basit
    
    keras.utils.to_categorical(tensor_ismi) -> direkt OHE veriyor
    
"""

model = CNN_classifier()
model.compile(optimizer = keras.optimizers.legacy.SGD(learning_rate=0.0001),
              loss = keras.losses.sparse_categorical_crossentropy,
              metrics = ["accuracy"])
history = model.fit(x = X_train, y=y_train, validation_data=(X_val, y_val),
          epochs=1, verbose=True)


model.evaluate(X_test, y_test) # evalution on test set with the metrics

#ayrica predictino yapma

preds = model.predict(X_test[:10])
print(preds) # probability leri dondurur (softmax ciktisi)


"""
Model.fit methodu History objesi donduryor,

onemli fonksyonalri

hitory.history -> dictionary which contains loss and extra metrids it measured 
hitory.params -> training paramters
history.epoch -> how many epochs
"""
print(history.history) # {'loss': [2.140880584716797], 'accuracy': [0.2782599925994873], 'val_loss': [1.9886666536331177], 'val_accuracy': [0.34540000557899475]}



