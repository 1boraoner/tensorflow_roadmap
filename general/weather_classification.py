import os
import warnings

from sklearn.exceptions import UndefinedMetricWarning

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, ReLU, Softmax, Dropout
from tensorflow.keras.optimizers.legacy import SGD, Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, AUC
from tensorflow.keras.callbacks import EarlyStopping


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, auc, confusion_matrix, classification_report


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
X_train, X_test, y_train, y_test = train_test_split(data, label_ohe, test_size=0.3)

# Model

def WeatherClassifier(num_features, dnn_sizes, num_classes:int):
    input_layer = keras.layers.Input(shape=(num_features))
    dnn_1 = Dense(units=dnn_sizes[0], activation='tanh')(input_layer)
    dropout_1 = Dropout(rate=0.1)(dnn_1)
    dnn_2 = Dense(units=dnn_sizes[1], activation='tanh')(dropout_1)
    dropout_2 = Dropout(rate=0.2)(dnn_2)
    dnn_3 = Dense(units=dnn_sizes[2], activation='tanh')(dropout_2)
    output = Dense(units=num_classes, activation="softmax")(dnn_3)
    return Model(inputs=input_layer, outputs=output)


early_stopper = EarlyStopping(monitor='loss', patience = 10, verbose = 1)

model = WeatherClassifier(num_features=X_train.shape[-1], dnn_sizes=[64, 64, 64],
                          num_classes=len(pd.unique(label)))

#training
model.compile(optimizer = Adam(learning_rate = 1.e-4),
              loss = CategoricalCrossentropy(from_logits=False),
              metrics = [AUC(), CategoricalAccuracy()])


model.fit(x=X_train, y=y_train, epochs=500, verbose=True,
          callbacks=[early_stopper])

preds = model.predict(X_test)
preds = tf.argmax(preds, axis=-1).numpy()

y_true = np.argmax(y_test, axis=-1)
conf_mat = confusion_matrix(y_true, preds)
acc = accuracy_score(y_true, preds)

print(label_enc.classes_)
print(conf_mat)
print(f"Test Accuracy: {acc}")
print(classification_report(preds, y_true))
