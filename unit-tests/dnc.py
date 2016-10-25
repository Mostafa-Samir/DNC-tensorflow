import tensorflow as tf
import numpy as np
import unittest

from dnc.dnc import DNC
from dnc.memory import Memory
from dnc.controller import BaseController

class DummyController(BaseController):
    def network(self, X):
        self.W = tf.Variable(tf.truncated_normal([self.nn_input_size, 64]))
        self.b = tf.Variable(tf.zeros([64]))

        return tf.matmul(X, self.W) + self.b


class DNCTest(unittest.TestCase):

    def test_construction(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as session:

                computer = DNC(DummyController, 10, 20, 10, 64, 1)

                self.assertEqual(computer.input_size, 10)
                self.assertEqual(computer.output_size, 20)
                self.assertEqual(computer.words_num, 10)
                self.assertEqual(computer.word_size, 64)
                self.assertEqual(computer.read_heads, 1)
                self.assertEqual(computer.batch_size, 1)

                self.assertTrue(isinstance(computer.memory, Memory))
                self.assertTrue(isinstance(computer.controller, DummyController))


    def test_call(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as session:

                computer = DNC(DummyController, 10, 20, 10, 64, 2, batch_size=3)
                input_batches = np.random.uniform(0, 1, (3, 5, 10)).astype(np.float32)

                input_tensor = tf.convert_to_tensor(input_batches)
                out_op, view_op = computer(input_tensor)

                session.run(tf.initialize_all_variables())
                out, view = session.run([out_op, view_op])

                self.assertEqual(out.shape, (3, 5, 20))
                self.assertEqual(view['free_gates'].shape, (3, 5, 2))
                self.assertEqual(view['allocation_gates'].shape, (3, 5, 1))
                self.assertEqual(view['write_gates'].shape, (3, 5, 1))
                self.assertEqual(view['read_weightings'].shape, (3, 5, 10, 2))
                self.assertEqual(view['write_weightings'].shape, (3, 5, 10))

if __name__ == '__main__':
    unittest.main(verbosity=2)
