import tensorflow as tf
import numpy as np
import unittest

from dnc.controller import BaseController

class DummyController(BaseController):
    def network(self, X):
        self.W = tf.Variable(tf.truncated_normal([self.nn_input_size, 64]))
        self.b = tf.Variable(tf.zeros([64]))

        return tf.matmul(X, self.W) + self.b


class DNCControllerTest(unittest.TestCase):

    def test_construction(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as session:

                controller = DummyController(10, 10, 2, 5)

                self.assertEqual(controller.nn_input_size, 20)
                self.assertEqual(controller.interface_vector_size, 38)
                self.assertEqual(controller.interface_weights.get_shape().as_list(), [64, 38])
                self.assertEqual(controller.nn_output_weights.get_shape().as_list(), [64, 10])
                self.assertEqual(controller.mem_output_weights.get_shape().as_list(), [10, 10])


    def test_get_nn_output_size(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as Session:

                controller = DummyController(10, 10, 2, 5)

                self.assertEqual(controller.get_nn_output_size(), 64)


    def test_parse_interface_vector(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as session:

                controller = DummyController(10, 10, 2, 5)
                zeta = np.random.uniform(-2, 2, (2, 38)).astype(np.float32)

                read_keys = np.reshape(zeta[:, :10], (-1, 5, 2))
                read_strengths = 1 + np.log(np.exp(np.reshape(zeta[:, 10:12], (-1, 2, ))) + 1)
                write_key = np.reshape(zeta[:, 12:17], (-1, 5))
                write_strength = 1 + np.log(np.exp(np.reshape(zeta[:, 17], (-1, 1, ))) + 1)
                erase_vector = 1.0 / (1 + np.exp(-1 * np.reshape(zeta[:, 18:23], (-1, 5))))
                write_vector = np.reshape(zeta[:, 23:28], (-1, 5))
                free_gates = 1.0 / (1 + np.exp(-1 * np.reshape(zeta[:, 28:30], (-1, 2))))
                allocation_gate = 1.0 / (1 + np.exp(-1 * zeta[:, 30]))
                write_gate = 1.0 / (1 + np.exp(-1 * zeta[:, 31]))
                read_modes = np.reshape(zeta[:, 32:], (-1, 3, 2))

                read_modes = np.transpose(read_modes, [0, 2, 1])
                read_modes = np.reshape(read_modes, (-1, 3))
                read_modes = np.exp(read_modes) / np.sum(np.exp(read_modes), axis=-1, keepdims=True)
                read_modes = np.reshape(read_modes, (2, 2, 3))
                read_modes = np.transpose(read_modes, [0, 2, 1])

                op = controller.parse_interface_vector(zeta)
                session.run(tf.initialize_all_variables())
                parsed = session.run(op)

                self.assertTrue(np.allclose(parsed['read_keys'], read_keys))
                self.assertTrue(np.allclose(parsed['read_strengths'], read_strengths))
                self.assertTrue(np.allclose(parsed['write_key'], write_key))
                self.assertTrue(np.allclose(parsed['write_strength'], write_strength))
                self.assertTrue(np.allclose(parsed['erase_vector'], erase_vector))
                self.assertTrue(np.allclose(parsed['write_vector'], write_vector))
                self.assertTrue(np.allclose(parsed['free_gates'], free_gates))
                self.assertTrue(np.allclose(parsed['allocation_gate'], allocation_gate))
                self.assertTrue(np.allclose(parsed['write_gate'], write_gate))
                self.assertTrue(np.allclose(parsed['read_modes'], read_modes))


    def test_process_input(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as session:

                controller = DummyController(10, 10, 2, 5)
                input_batch = np.random.uniform(0, 1, (2, 10)).astype(np.float32)
                last_read_vectors = np.random.uniform(-1, 1, (2, 5, 2)).astype(np.float32)

                v_op, zeta_op = controller.process_input(input_batch, last_read_vectors)

                session.run(tf.initialize_all_variables())
                v, zeta = session.run([v_op, zeta_op])

                self.assertEqual(v.shape, (2, 10))
                self.assertEqual(np.concatenate([np.reshape(val, (2, -1)) for _,val in zeta.iteritems()], axis=1).shape, (2, 38))


    def test_final_output(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as session:

                controller = DummyController(10, 10, 2, 5)
                output_batch = np.random.uniform(0, 1, (2, 10)).astype(np.float32)
                new_read_vectors = np.random.uniform(-1, 1, (2, 5, 2)).astype(np.float32)

                op = controller.final_output(output_batch, new_read_vectors)
                session.run(tf.initialize_all_variables())
                y = session.run(op)

                self.assertEqual(y.shape, (2, 10))


if __name__ == '__main__':
    unittest.main(verbosity=2)
