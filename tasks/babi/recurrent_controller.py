import numpy as np
import tensorflow as tf
from dnc.controller import BaseController

"""
A 1-layer LSTM recurrent neural network with 256 hidden units
Note: the state of the LSTM is not saved in a variable because we want
the state to reset to zero on every input sequence
"""


class RecurrentController(BaseController):

    def network_vars(self):
        self.lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(256)
        self.state = self.lstm_cell.zero_state(self.batch_size, tf.float32)

    def network_op(self, X, state):
        X = tf.convert_to_tensor(X)
        return self.lstm_cell(X, state)

    def get_state(self):
        return self.state

    @staticmethod
    def update_state(new_state):
        return tf.no_op()
