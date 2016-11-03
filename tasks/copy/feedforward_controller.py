import numpy as np
import tensorflow as tf
from dnc.controller import BaseController


"""
A 2-Layers feedforward neural network with 128, 256 nodes respectively
"""

class FeedforwardController(BaseController):

    def network_vars(self):
        initial_std = lambda in_nodes: np.min(1e-4, np.sqrt(2.0 / in_nodes))
        input_ = self.nn_input_size

        self.W1 = tf.Variable(tf.truncated_normal([input_, 128], stddev=initial_std(input_)))
        self.W2 = tf.Variable(tf.truncated_normal([128, 256], stddev=initial_std(128)))
        self.b1 = tf.Variable(tf.zeros([128]))
        self.b2 = tf.Variable(tf.zeros([256]))


    def network_op(self, X):
        layer1_activation = tf.nn.relu(tf.matmul(X, self.W1) + self.b1)
        layer2_activation = tf.nn.relu(tf.matmul(layer1_activation, self.W2) + self.b2)

        return layer2_activation
