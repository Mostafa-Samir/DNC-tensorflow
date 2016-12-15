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


    def test_unpack_into_tensorarray(self):
         graph = tf.Graph()
         with graph.as_default():
             with tf.Session(graph=graph) as session:

                 T = tf.random_normal([5, 10, 7, 7])
                 ta = util.unpack_into_tensorarray(T, axis=1)

                 vT, vTA5 = session.run([T, ta.read(5)])

                 self.assertEqual(vTA5.shape, (5, 7, 7))
                 self.assertTrue(np.allclose(vT[:, 5, :, :], vTA5))


    def test_pack_into_tensor(self):
         graph = tf.Graph()
         with graph.as_default():
             with tf.Session(graph=graph) as session:

                T = tf.random_normal([5, 10, 7, 7])
                ta = util.unpack_into_tensorarray(T, axis=1)
                pT = util.pack_into_tensor(ta, axis=1)

                vT, vPT = session.run([T, pT])

                self.assertEqual(vPT.shape, (5, 10, 7, 7))
                self.assertTrue(np.allclose(vT, vPT)) 


if __name__ == "__main__":
    unittest.main(verbosity=2)
