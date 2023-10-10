import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

""""
    Tensorflow operations revolve around tensors which flow operation to operation
    A tensor is very similar to NumPy ndarray.
    it can be multidimensinal array or a scalar.
"""

#Tensors and Operations

scalar_tensor = tf.constant([42])
print(scalar_tensor)

tensor_ex = tf.constant([[1,2,3],[4,5,6]])
print(tensor_ex)
tensor_shape = tensor_ex.shape
tensor_dtype = tensor_ex.dtype

#indexing works much like in numpy
print(tensor_ex[:, 1:])
print(tensor_ex[:, 1, tf.newaxis])

#all sorts of tensor operations are available

sqrt = tf.sqrt(tf.cast(tensor_ex, tf.float32))
print(sqrt)

G = tensor_ex @ tf.transpose(tensor_ex, perm=[1, 0])
print(G)

# some hidden operations also call tf. ops like addition
# tf.add(t, 10) === t + 10
# @ operations is shortcut for tf.matmul

# some takeways in difference between Numpy and Tensorflow

#tf.transpose() is used rather than A.T like in numpy because tf.trasnpose generate a new tensor careted
#whereas numpy.T is a view of the array's transpose

# similar to tf.reduce_mean() as np.mean()

a = np.array([2., 4., 5.])
b = tf.constant(a)
c = tf.numpy()


#custome loss function

# HuberLoss

def huber_loss(y_true, y_pred):
    error = y_true - y_pred
    is_small_error = tf.abs(error) < 1
    squared_loss = tf.square(error) / 2
    linear_loss = tf.abs(error) - 0.5
    return tf.where(is_small_error, squared_loss, linear_loss)

# model.compile(loss=huber_loss, optimizer=keras.optimizers.Adam())
# model.fit(X_train, y_train, [...])

class HuberLoss(keras.losses.Loss):
    def __init__(self, threshold=1.0, **kwargs):
        super(HuberLoss, self).__init__(**kwargs)
        self.threshold = threshold

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < self.threshold
        squared_loss = tf.square(error) / 2
        linear_loss = tf.abs(error) - 0.5
        return tf.where(is_small_error, squared_loss, linear_loss)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold":self.threshold}

#model.compile(loss=HuberLoss(2.0), optimziers="nadam")

# Custom Keras Functions

def my_softplus(z):
    return tf.math.log(tf.exp(z) + 1.0)

def my_golorot_initializer(shape, dtype=tf.float32):
    stddev = tf.sqrt(2.0 / (shape[0]+shape[1]))
    return tf.random.normal(shape, stddev=stddev, dtype=dtype)

def my_l1_regularizer(weights):
    return tf.reduce_sum(tf.abs(0.001 * weights))


layer = keras.layers.Dense(units=30, activation =my_softplus,
                           kernel_initializers=my_golorot_initializer,
                           kernel_regularizers=my_l1_regularizer)

# tabii yine model save edilmek istenirse bunlarin hepsi subclass yazilmasi gerekmekte

def MyL1Regularizer(keras.regularizers.Regularizer):
    def __init__(self, lambda_param, **kwargs):
        self.factor = lambda_param
        super(MyL1Regularizer,self).__init__(**kwargs)
    def __call__(self, weights):
        return tf.reduce_sum(tf.abs(self.factor*weights))
    def get_config(self):
        return {"factor":self.factor}



# custom metrics

class HuberMetric(keras.metrics.Metric):
    def __init__(self, threshold=1.0, **kwargs):
        super(HuberMetric, self).__init__(**kwargs)
        self.threshold = threshold
        self.huber_fn = create_huber(threshold)
        self.total = self.add_weight("total", initializers="zeros")
        self.count = self.add_weight("count", initializers="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        metric = self.huber_fn(y_true, y_pred)
        self.total.assign_add(tf.reduce_sum(metric))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))
    def result(self):
        return self.total / self.count
    def get_config(self):
        base_config = super().get_config()
        return {base_config, "threshold":self.threshold}


# Jaccard Similarity Metric

class JaccardSimilarity(keras.metrics.Metric):
    def __init__(self):
        super(JaccardSimilarity, self).__init__(**kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        inters = y_true * y_pred
        unions = (y_true + y_pred) - inters
        jaccard_similarity = tf.reduce_sum(inters, axis=-1) / tf.reduce_sum(unions, axis=-1)
        self.total.assign_add(tf.reduce_sum(jaccard_similarity))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))
    def results(self):
        return self.total / self.count
    def get_config(self):
        base_config = super().get_config()
        return {base_config}


#Custom Layers

class MyDense(keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
    def build(self, batch_input_shape):
        self.kernel = self.add_weight(
            name="Kernel", shape=[batch_input_shape[-1], self.units],
            initializers="glorot_normal")
        self.bias = self.add_weight(
            name="bias", shape=[self.units], initializers= "zeros"
        )
        super().build(batch_input_shape) # must be at the end
    def call(self, X):
        return self.activation(X @ self.kernel + self.bias)
    def compute_output_shape(self, batch_input_shape):
        return tf.TensorShape(batch_input_shape.as_list()[:-1] + [self.units])
    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                "units":self.units,
                "activations":keras.activations.serialize(self.activation)}



class MyMultiLayer(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(MyMultiLayer, self).__init__(**kwargs)

    def call(self, X): # returns three different outputs
        X1, X2 = X
        return [X1 + X2, X1* X2, X1/X2]

    def compute_output_shape(self, batch_input_shape): # for each output the output shapes are given in list
        b1, b2 = batch_input_shape
        return [b1,b1,b1]



# custom layer where acts differntly on training and testing
class GaussianNoise(keras.layers.Layer):
    def __init__(self, stddev, **kwargs):
        super(GaussianNoise, self).__init__(**kwargs)
        self.stddev = stddev
    def call(self, X):
        if training:
            noise = tf.random.normal(tf.shape(X), stddev = self.stddev)
            return  X + noise
        else:
            return X
    def compute_output_shape(self, batch_input_shape):
        return batch_input_shape



class ResidualBlock(keras.layers.Layer):

    def __init__(self, n_layers, units, activations, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.units = units
        self.activations = [keras.activations.get(activation) for activation in activations]

        self.hidden = [keras.layers.Dense(unit=units[i], activation=activations[i]) for i in range(n_layers)]
    def call(self, inputs):
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
        return inputs + Z
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "units":units, "activations":keras.activations.serialize(activations)}
#subclassing API
class ResidualRegressor(keras.Model):

    def __init__(self, output_dim, **kwargs):
        super(ResidualRegressor, self).__init__(**kwargs)
        self.hidden1 = keras.layers.Dense(30, activation="elu", kernel_initializer="he_normal")
        self.block1 = ResidualBlock(2, 30)
        self.block2 = ResidualBlock(2, 30)
        self.out = keras.layers.Dense(output_dim)

    def call(self, inputs):
        Z = self.hidden1(inputs)
        for _ in range(3):
            Z = self.block1(Z)
        Z = self.block2
        return self.out(Z)



