import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings(action='ignore', category=FutureWarning)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, ReLU, Softmax, Dropout
from tensorflow.keras.optimizers.legacy import SGD
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, AUC

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix


data = pd.read_csv("../datasets/weather_prediction/seattle-weather.csv")
data, raw_label = data.iloc[:, :-1], data.iloc[:, -1]

for c in pd.unique(raw_label):
    print(f"{c}: ", (raw_label==c).sum())

data[['precipitation']] = data[['precipitation']].fillna(value=data[['precipitation']].mean())
data[['temp_max']] = data[['temp_max']].fillna(value=data[['temp_max']].mean())
data[['temp_min']] = data[['temp_min']].fillna(value=data[['temp_min']].mean())
data[['wind']] = data[['wind']].fillna(value=data[['wind']].mean())

# Date are split and one hot encoded

data["date"] = pd.to_datetime(data["date"])
data["month"] = data["date"].dt.month
data["day"] = data["date"].dt.day
data.drop(inplace=True, columns=["date"])

encoder = OneHotEncoder(sparse=False)
encoder.fit(data[['month', 'day']])
encoded_month_day = encoder.transform(data[['month', 'day']])
data.drop(inplace=True, columns=["month"])
data.drop(inplace=True, columns=["day"])

data = np.asarray(data)
date_data = np.asarray(encoded_month_day)
data = np.concatenate([data, date_data], axis=-1)

# Label Encoded
label_enc = LabelEncoder()
label = label_enc.fit_transform(raw_label)
print(label_enc.classes_)
label_ohe = np.eye(label.shape[0], len(pd.unique(label)))[label]
X_train, X_test, y_train, y_test = train_test_split(data, label_ohe, test_size=0.20)

# Model

def WeatherClassifier(num_features, dnn_sizes, num_classes:int):
    input_layer = keras.layers.Input(shape=(num_features))
    dnn_1 = Dense(units=dnn_sizes[0], activation='relu')(input_layer)
    dropout_1 = Dropout(rate=0.1)(dnn_1)
    dnn_2 = Dense(units=dnn_sizes[1], activation='relu')(dropout_1)
    dnn_3 = Dense(units=dnn_sizes[2], activation='relu')(dnn_2)
    output = Dense(units=num_classes, activation="softmax")(dnn_3)
    return Model(inputs=input_layer, outputs=output)


model = WeatherClassifier(num_features=X_train.shape[-1], dnn_sizes=[64, 128, 64],
                          num_classes=len(pd.unique(label)))

#training
model.compile(optimizer = SGD(learning_rate = 1.e-3),
              loss = CategoricalCrossentropy(from_logits=False),
              metrics = [AUC(), CategoricalAccuracy()])

model.fit(x=X_train, y=y_train, epochs=100, verbose=True)

preds = model.predict(X_test)
preds = tf.argmax(preds, axis=-1)

conf_mat = confusion_matrix(np.argmax(y_test, axis=-1), preds.numpy())
print(conf_mat)