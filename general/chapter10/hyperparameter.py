import tensorflow as tf
from tensorflow import keras
from hyperopt import tpe, hp, fmin, Trials, STATUS_OK

(X_train, y_train), (X_test, y_test) = keras.datasets.boston_housing.load_data(path="../datasets/boston_housing.npz", test_split=0.25, seed=113)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)
def model_def(params):

    def simple_mlp(input_shape, dnn_units, hidden_layers):
        input_ = keras.layers.Input(input_shape)
        for i in range(hidden_layers):
            if i == 0:
                dnn_in = keras.layers.Dense(units=dnn_units, activation='relu')(input_)
                continue
            dnn_in = keras.layers.Dense(units=dnn_units,activation='relu')(dnn_in)
        output = keras.layers.Dense(units=1)(dnn_in)
        return keras.Model(inputs=input_, outputs=output)

    model = simple_mlp(input_shape=params['input_shape'], dnn_units=params['dnn_units'], hidden_layers=params["hidden_layers"])
    model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=params['learning_rate']),
                  loss = keras.losses.mean_squared_error)

    history = model.fit(x = X_train, y = y_train, validation_split=0.1, epochs=300,
                        callbacks= keras.callbacks.EarlyStopping(patience=10), verbose=0)
    return {"loss": min(history.history['val_loss']), "status": STATUS_OK}

space = {
    "input_shape": X_train.shape[-1],
    "dnn_units": hp.choice("dnn_units", list(range(32, 256, 32))),
    "learning_rate": hp.choice("learning_rate", [0.001, 0.0001]),
    "hidden_layers": hp.choice("hidden_layers", [1, 2, 3, 4])
}

trials = Trials()
best = fmin(model_def, space, algo=tpe.suggest, max_evals=100, trials=trials,return_argmin=False)
print('best: ', best)


