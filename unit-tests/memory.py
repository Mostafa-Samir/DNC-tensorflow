import tensorflow as tf
import numpy as np
import unittest

from dnc.memory import Memory

class DNCMemoryTests(unittest.TestCase):

    def test_construction(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as session:

                mem = Memory(4, 5, 2)
                session.run(tf.initialize_all_variables())

                self.assertEqual(mem.words_num, 4)
                self.assertEqual(mem.word_size, 5)
                self.assertEqual(mem.read_heads, 2)

                self.assertTrue(isinstance(mem.memory_matrix, tf.Variable))
                self.assertEqual(mem.memory_matrix.get_shape().as_list(), [4, 5])
                self.assertTrue(isinstance(mem.usage_vector, tf.Variable))
                self.assertEqual(mem.usage_vector.get_shape().as_list(), [4])
                self.assertTrue(isinstance(mem.link_matrix, tf.Variable))
                self.assertEqual(mem.link_matrix.get_shape().as_list(), [4, 4])
                self.assertTrue(isinstance(mem.write_weighting, tf.Variable))
                self.assertEqual(mem.write_weighting.get_shape().as_list(), [4])
                self.assertTrue(isinstance(mem.read_weightings, tf.Variable))
                self.assertEqual(mem.read_weightings.get_shape().as_list(), [2, 4])
                self.assertTrue(isinstance(mem.read_vectors, tf.Variable))
                self.assertEqual(mem.read_vectors.get_shape().as_list(), [2, 5])


    def test_lookup_weighting(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as session:

                mem = Memory(4, 5, 2)
                keys = np.array([[0., 1., 0., 0.3, 4.3], [1.3, 0.8, 0., 0., 0.62]]).astype(np.float32)
                strengths = np.array([0.7, 0.2]).astype(np.float32)
                predicted_weights = np.array([[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]]).astype(np.float32)

                op = mem.get_lookup_weighting(keys, strengths)
                session.run(tf.initialize_all_variables())
                c = session.run(op)

                self.assertTrue(c.shape, (2, 4))
                self.assertTrue(np.array_equal(c, predicted_weights))


    def test_update_usage_vector(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as session:

                mem = Memory(4, 5, 2)
                free_gates = np.array([0.2, 0.67]).astype(np.float32)
                predicted_usage = np.array([0.49429685,  0.49429685,  0.49429685,  0.49429685]).astype(np.float32)

                changes = [
                    mem.read_weightings.assign(tf.fill([2, 4], 0.25)),
                    mem.write_weighting.assign(tf.fill([4, ], 0.25)),
                    mem.usage_vector.assign(tf.fill([4, ], 0.5))
                ]
                op = mem.update_usage_vector(free_gates)
                session.run(tf.initialize_all_variables())
                session.run(changes)
                u = session.run(op)
                updated_usage = session.run(mem.usage_vector.value())

                self.assertEqual(u.shape, (4, ))
                self.assertTrue(np.array_equal(u, predicted_usage))
                self.assertTrue(np.array_equal(updated_usage, predicted_usage))


    def test_get_allocation_weighting(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as session:

                mem = Memory(4, 5, 2)
                sorted_usage = np.array([0.2, 0.4, 0.7, 1]).astype(np.float32)
                free_list = np.array([2, 3, 1, 0]).astype(np.int32)
                predicted_weights = np.array([0., 0.024, 0.8, 0.12]).astype(np.float32)

                op = mem.get_allocation_weighting(sorted_usage, free_list)
                session.run(tf.initialize_all_variables())
                a = session.run(op)

                self.assertEqual(a.shape, (4, ))
                self.assertTrue(np.allclose(a, predicted_weights))


    def test_updated_write_weighting(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as session:

                mem = Memory(4, 5, 2)
                write_gate, allocation_gate = 0.65, 0.2
                lookup_weighting = np.array([[0.25, 0.25, 0.25, 0.25]]).astype(np.float32)
                allocation_weighting = np.array([0., 0.024, 0.8, 0.12]).astype(np.float32)
                predicted_weights = np.array([0.13, 0.13312, 0.234, 0.14560001]).astype(np.float32)

                op = mem.update_write_weighting(lookup_weighting, allocation_weighting, write_gate, allocation_gate)
                session.run(tf.initialize_all_variables())
                w_w = session.run(op)
                updated_write_weighting = session.run(mem.write_weighting.value())

                self.assertEqual(w_w.shape, (4, ))
                self.assertTrue(np.allclose(w_w, predicted_weights))
                self.assertTrue(np.allclose(updated_write_weighting, predicted_weights))


    def test_update_memory(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as session:

                mem = Memory(4, 5, 2)
                write_weighting = np.array([0.13, 0.13312, 0.234, 0.14560001]).astype(np.float32)
                write_vector = np.array([1.8, 3.548, 4.2, 0.269, 0.001]).astype(np.float32)
                erase_vector = np.zeros(5).astype(np.float32)
                predicted = np.outer(write_weighting, write_vector)

                op = mem.update_memory(write_weighting, write_vector, erase_vector)
                session.run(tf.initialize_all_variables())
                M = session.run(op)
                updated_memory = session.run(mem.memory_matrix.value())

                self.assertEqual(M.shape, (4, 5))
                self.assertTrue(np.allclose(M, predicted))
                self.assertTrue(np.allclose(updated_memory, predicted))

    def test_update_precedence_vector(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as session:

                mem = Memory(4, 5, 2)
                write_weighting = np.array([0.13, 0.13312, 0.234, 0.14560001]).astype(np.float32)
                predicted = write_weighting

                op = mem.update_precedence_vector(write_weighting)
                session.run(tf.initialize_all_variables())
                p = session.run(op)
                updated_precedence_vector = session.run(mem.precedence_vector.value())

                self.assertEqual(p.shape, (4, ))
                self.assertTrue(np.allclose(p, predicted))
                self.assertTrue(np.allclose(updated_precedence_vector, predicted))


    def test_update_link_matrix(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as session:

                mem = Memory(4, 5, 2)
                _write_weighting = np.array([0.13, 0.13312, 0.234, 0.14560001]).astype(np.float32)
                _precedence_vector = np.array([0.17644639, 0.18068111, 0.31760353, 0.19761997]).astype(np.float32)
                initial_link = np.random.uniform(0, 1, (4, 4)).astype(np.float32)
                np.fill_diagonal(initial_link, 0)

                # calculate the updated link iteratively as in paper
                # to check the correcteness of the vectorized implementation
                predicted = np.zeros((4,4), dtype=np.float32)
                for i in range(4):
                    for j in range(4):
                        if i != j:
                            reset_factor = (1 - _write_weighting[i] - _write_weighting[j])
                            predicted[i, j]  = reset_factor * initial_link[i , j] + _write_weighting[i] * _precedence_vector[j]

                changes = [
                    mem.link_matrix.assign(initial_link),
                    mem.precedence_vector.assign(_precedence_vector)
                ]

                write_weighting = tf.constant(_write_weighting)

                op = mem.update_link_matrix(write_weighting)
                session.run(tf.initialize_all_variables())
                session.run(changes)
                L = session.run(op)
                updated_link_matrix = session.run(mem.link_matrix.value())

                self.assertTrue(np.allclose(L, predicted))
                self.assertTrue(np.allclose(updated_link_matrix, predicted))


    def test_get_directional_weightings(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as session:

                mem = Memory(4, 5, 2)
                _link_matrix = np.random.uniform(0, 1, (4, 4)).astype(np.float32)
                _read_weightings = np.full((2, 4), 0.25)
                predicted_forward = np.dot(_read_weightings, _link_matrix)
                predicted_backward = np.dot(_read_weightings, _link_matrix.T)

                changes = [
                    mem.read_weightings.assign(_read_weightings)
                ]

                fop, bop = mem.get_directional_weightings(_link_matrix)

                session.run(tf.initialize_all_variables())
                session.run(changes)
                forward_weighting, backward_weighting = session.run([fop, bop])

                self.assertTrue(np.allclose(forward_weighting, predicted_forward))
                self.assertTrue(np.allclose(backward_weighting, predicted_backward))



    def test_update_read_weightings(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as session:

                mem = Memory(4, 5, 2)
                lookup_weightings = np.full((2, 4), 0.25).astype(np.float32)
                forward_weighting = np.array([[0.5, 0.23, 0.14, 0.062], [0.062, 0.23, 0.5, 0.14]]).astype(np.float32)
                backward_weighting = np.array([[0.01, 0.15, 0.068, 0.62], [0.62, 0.15, 0.01, 0.62]]).astype(np.float32)
                read_mode = np.array([[0.1, 0.6, 0.3], [0.01, 0.98, 0.01]]).astype(np.float32)
                predicted_weights = np.zeros((2, 4)).astype(np.float32)

                # calculate the predicted weights using iterative method from paper
                # to check the correcteness of the vectorized implementation
                for i in range(2):
                    predicted_weights[i] = read_mode[i,0] * backward_weighting[i] + read_mode[i, 1] * lookup_weightings[i] + read_mode[i, 2] * forward_weighting[i]

                op = mem.update_read_weightings(lookup_weightings, forward_weighting, backward_weighting, read_mode)
                session.run(tf.initialize_all_variables())
                w_r = session.run(op)
                updated_read_weightings = session.run(mem.read_weightings.value())

                self.assertTrue(np.allclose(w_r, predicted_weights))
                self.assertTrue(np.allclose(updated_read_weightings, predicted_weights))


    def test_update_read_vectors(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph = graph) as session:

                mem = Memory(4, 5, 2)
                memory_matrix = np.random.uniform(-1, 1, (4, 5)).astype(np.float32)
                read_weightings = np.array([[0.07, 0.36, 0.51, 0.06], [0.51, 0.07, 0.06, 0.36]]).astype(np.float32)
                predicted = np.dot(read_weightings, memory_matrix)

                op = mem.update_read_vectors(memory_matrix, read_weightings)
                session.run(tf.initialize_all_variables())
                r = session.run(op)
                updated_read_vectors = session.run(mem.read_vectors.value())

                self.assertTrue(np.allclose(r, predicted))
                self.assertTrue(np.allclose(updated_read_vectors, predicted))


if __name__ == '__main__':
    unittest.main(verbosity=2)
