@bora.oner

"""
    Jaccard Similarity Metric Tensorflow Implementation 
"""

import tensorflow as tf
import tensorflow.keras as keras

class JaccardSimilarity(keras.metrics.Metric):
    def __init__(self, name="JaccardSimilarity", **kwargs):
        super(JaccardSimilarity, self).__init__(name=name, **kwargs)
        self.scores = self.add_weight(name='jaccard_scores', initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, dtype=tf.int8)
        top_k_values = tf.math.top_k(y_pred, k=3, sorted=True)
        less_than_kth = y_pred >= tf.expand_dims(top_k_values[0][:, -1], axis=-1)
        less_than_kth = tf.cast(less_than_kth, dtype=tf.int8)
        product = tf.multiply(y_true, less_than_kth)
        intersections = tf.reduce_sum(product, axis=-1)
        union = tf.reduce_sum(((y_true + less_than_kth) - product), axis=-1)
        jaccard = tf.reduce_mean(intersections / union)
        self.scores.assign_add(tf.reduce_sum(jaccard))
        self.count.assign_add(1)

    def result(self):
        return self.scores / self.count
    def reset_state(self):
        self.scores.assign(0)
        self.count.assign(0)


# Example Use
# in keras.Model.compile(metrics=[JaccardSimilarity()])
