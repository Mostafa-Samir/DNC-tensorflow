import numpy as np
import tensorflow as tf
from dnc.controller import BaseController


"""
A 2-Layers feedforward neural network with 128, 256 nodes respectively
"""

class FeedforwardController(BaseController):

    def network_vars(self):
        initial_std = lambda in_nodes: np.minimum(1e-2, np.sqrt(2.0 / in_nodes))
        input_ = self.nn_input_size

        self.W1 = tf.Variable(tf.truncated_normal([input_, 128], stddev=initial_std(input_)), name='layer1_W')
        self.W2 = tf.Variable(tf.truncated_normal([128, 256], stddev=initial_std(128)), name='layer2_W')
        self.b1 = tf.Variable(tf.zeros([128]), name='layer1_b')
        self.b2 = tf.Variable(tf.zeros([256]), name='layer2_b')


    def network_op(self, X):
        l1_output = tf.matmul(X, self.W1) + self.b1
        l1_activation = tf.nn.relu(l1_output)

        l2_output = tf.matmul(l1_activation, self.W2) + self.b2
        l2_activation = tf.nn.relu(l2_output)

        return l2_activation

    def initials(self):
        initial_std = lambda in_nodes: np.minimum(1e-2, np.sqrt(2.0 / in_nodes))

        # defining internal weights of the controller
        self.interface_weights = tf.Variable(
            tf.truncated_normal([self.nn_output_size, self.interface_vector_size], stddev=initial_std(self.nn_output_size)),
            name='interface_weights'
        )
        self.nn_output_weights = tf.Variable(
            tf.truncated_normal([self.nn_output_size, self.output_size], stddev=initial_std(self.nn_output_size)),
            name='nn_output_weights'
        )
        self.mem_output_weights = tf.Variable(
            tf.truncated_normal([self.word_size * self.read_heads, self.output_size],  stddev=initial_std(self.word_size * self.read_heads)),
            name='mem_output_weights'
        )
