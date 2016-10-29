import tensorflow as tf
import numpy as np
import unittest

import dnc.utility as util

class DNCUtilityTests(unittest.TestCase):

    def test_pairwise_add(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as session:

                _u = np.array([5, 6])
                _v = np.array([1, 2])

                predicted_U = np.array([[10, 11], [11, 12]])
                predicted_UV = np.array([[6, 7], [7, 8]])

                u = tf.constant(_u)
                v = tf.constant(_v)

                U_op = util.pairwise_add(u)
                UV_op = util.pairwise_add(u, v)

                U, UV = session.run([U_op, UV_op])

                self.assertTrue(np.allclose(U, predicted_U))
                self.assertTrue(np.allclose(UV, predicted_UV))


    def test_pairwise_add_with_batch(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as session:

                _u = np.array([[5, 6], [7, 8]])
                _v = np.array([[1, 2], [3, 4]])

                predicted_U = np.array([[[10, 11], [11, 12]], [[14, 15], [15, 16]]])
                predicted_UV = np.array([[[6, 7], [7, 8]], [[10, 11], [11, 12]]])

                u = tf.constant(_u)
                v = tf.constant(_v)

                U_op = util.pairwise_add(u, is_batch=True)
                UV_op = util.pairwise_add(u, v, is_batch=True)

                U, UV = session.run([U_op, UV_op])

                self.assertTrue(np.allclose(U, predicted_U))
                self.assertTrue(np.allclose(UV, predicted_UV))


    def test_allocate(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as session:

                var, var_name = util.allocate([4])
                indecies = np.array([0, 1, 2, 3])
                updates = np.array([7, 8, 9, 10], dtype=np.float32)

                update = tf.scatter_update(var, indecies, updates)
                val = session.run(update)

                self.assertTrue(np.allclose(val, updates))


    def test_read_and_deallocate(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as session:

                var, var_name = util.allocate([4])
                indecies = np.array([0, 1, 2, 3])
                updates = np.array([7, 8, 9, 10], dtype=np.float32)

                update_op = tf.scatter_update(var, indecies, updates)
                with tf.control_dependencies([update_op]):
                    da_op = util.read_and_deallocate(var, var_name)

                    val = session.run(da_op)
                    self.assertTrue(np.allclose(val, updates))

if __name__ == "__main__":
    unittest.main(verbosity=2)
