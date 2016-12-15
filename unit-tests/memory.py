import tensorflow as tf
import numpy as np
import unittest

from dnc.memory import Memory

def random_softmax(shape, axis):
    rand = np.random.uniform(0, 1, shape).astype(np.float32)
    return np.exp(rand) / np.sum(np.exp(rand), axis=axis, keepdims=True)

class DNCMemoryTests(unittest.TestCase):

    def test_construction(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as session:

                mem = Memory(4, 5, 2, 2)
                session.run(tf.initialize_all_variables())

                self.assertEqual(mem.words_num, 4)
                self.assertEqual(mem.word_size, 5)
                self.assertEqual(mem.read_heads, 2)
                self.assertEqual(mem.batch_size, 2)


    def test_init_memory(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as session:

                mem = Memory(4, 5, 2, 2)
                M, u, p, L, ww, rw, r = session.run(mem.init_memory())

                self.assertEqual(M.shape, (2, 4, 5))
                self.assertEqual(u.shape, (2, 4))
                self.assertEqual(L.shape, (2, 4, 4))
                self.assertEqual(ww.shape, (2, 4))
                self.assertEqual(rw.shape, (2, 4, 2))
                self.assertEqual(r.shape, (2, 5, 2))
                self.assertEqual(p.shape, (2, 4))

    def test_lookup_weighting(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as session:

                mem = Memory(4, 5, 2, 2)
                initial_mem = np.random.uniform(0, 1, (2, 4, 5)).astype(np.float32)
                keys = np.random.uniform(0, 1, (2, 5, 2)).astype(np.float32)
                strengths = np.random.uniform(0, 1, (2 ,2)).astype(np.float32)

                norm_mem = initial_mem / np.sqrt(np.sum(initial_mem ** 2, axis=2, keepdims=True))
                norm_keys = keys/ np.sqrt(np.sum(keys ** 2, axis = 1, keepdims=True))
                sim = np.matmul(norm_mem, norm_keys)
                sim = sim * strengths[:, np.newaxis, :]
                predicted_wieghts = np.exp(sim) / np.sum(np.exp(sim), axis=1, keepdims=True)

                memory_matrix = tf.convert_to_tensor(initial_mem)
                op = mem.get_lookup_weighting(memory_matrix, keys, strengths)
                c = session.run(op)

                self.assertEqual(c.shape, (2, 4, 2))
                self.assertTrue(np.allclose(c, predicted_wieghts))


    def test_update_usage_vector(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as session:

                mem = Memory(4, 5, 2, 2)
                free_gates = np.random.uniform(0, 1, (2, 2)).astype(np.float32)
                init_read_weightings = random_softmax((2, 4, 2), axis=1)
                init_write_weightings = random_softmax((2, 4), axis=1)
                init_usage = np.random.uniform(0, 1, (2, 4)).astype(np.float32)

                psi = np.product(1 - init_read_weightings * free_gates[:, np.newaxis, :], axis=2)
                predicted_usage = (init_usage + init_write_weightings - init_usage * init_write_weightings) * psi


                read_weightings = tf.convert_to_tensor(init_read_weightings)
                write_weighting = tf.convert_to_tensor(init_write_weightings)
                usage_vector = tf.convert_to_tensor(init_usage)

                op = mem.update_usage_vector(usage_vector, read_weightings, write_weighting, free_gates)
                u = session.run(op)

                self.assertEqual(u.shape, (2, 4))
                self.assertTrue(np.array_equal(u, predicted_usage))


    def test_get_allocation_weighting(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as session:

                mem = Memory(4, 5, 2, 2)
                mock_usage = np.random.uniform(0.01, 1, (2, 4)).astype(np.float32)
                sorted_usage = np.sort(mock_usage, axis=1)
                free_list = np.argsort(mock_usage, axis=1)

                predicted_weights = np.zeros((2, 4)).astype(np.float32)
                for i in range(2):
                    for j in range(4):
                        product_list = [mock_usage[i, free_list[i,k]] for k in range(j)]
                        predicted_weights[i, free_list[i,j]] = (1 - mock_usage[i, free_list[i, j]]) * np.product(product_list)

                op = mem.get_allocation_weighting(sorted_usage, free_list)
                a = session.run(op)

                self.assertEqual(a.shape, (2, 4))
                self.assertTrue(np.allclose(a, predicted_weights))


    def test_updated_write_weighting(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as session:

                mem = Memory(4, 5, 2, 2)
                write_gate = np.random.uniform(0, 1, (2,1)).astype(np.float32)
                allocation_gate = np.random.uniform(0, 1, (2,1)).astype(np.float32)
                lookup_weighting = random_softmax((2, 4, 1), axis=1)
                allocation_weighting = random_softmax((2, 4), axis=1)

                predicted_weights = write_gate * (allocation_gate * allocation_weighting + (1 - allocation_gate) * np.squeeze(lookup_weighting))

                op = mem.update_write_weighting(lookup_weighting, allocation_weighting, write_gate, allocation_gate)
                w_w = session.run(op)

                self.assertEqual(w_w.shape, (2,4))
                self.assertTrue(np.allclose(w_w, predicted_weights))


    def test_update_memory(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as session:

                mem = Memory(4, 5, 2, 2)
                write_weighting = random_softmax((2, 4), axis=1)
                write_vector = np.random.uniform(0, 1, (2, 5)).astype(np.float32)
                erase_vector = np.random.uniform(0, 1, (2, 5)).astype(np.float32)
                memory_matrix = np.random.uniform(-1, 1, (2, 4, 5)).astype(np.float32)

                ww = write_weighting[:, :, np.newaxis]
                v, e = write_vector[:, np.newaxis, :], erase_vector[:, np.newaxis, :]
                predicted = memory_matrix * (1 - np.matmul(ww, e)) + np.matmul(ww, v)

                memory_matrix = tf.convert_to_tensor(memory_matrix)

                op = mem.update_memory(memory_matrix, write_weighting, write_vector, erase_vector)
                M = session.run(op)

                self.assertEqual(M.shape, (2, 4, 5))
                self.assertTrue(np.allclose(M, predicted))

    def test_update_precedence_vector(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as session:

                mem = Memory(4, 5, 2, 2)
                write_weighting = random_softmax((2, 4), axis=1)
                initial_precedence = random_softmax((2, 4), axis=1)
                predicted = (1 - write_weighting.sum(axis=1, keepdims=True)) * initial_precedence + write_weighting

                precedence_vector = tf.convert_to_tensor(initial_precedence)

                op = mem.update_precedence_vector(precedence_vector, write_weighting)
                p = session.run(op)

                self.assertEqual(p.shape, (2,4))
                self.assertTrue(np.allclose(p, predicted))


    def test_update_link_matrix(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as session:

                mem = Memory(4, 5, 2, 2)
                _write_weighting = random_softmax((2, 4), axis=1)
                _precedence_vector = random_softmax((2, 4), axis=1)
                initial_link = np.random.uniform(0, 1, (2, 4, 4)).astype(np.float32)
                np.fill_diagonal(initial_link[0,:], 0)
                np.fill_diagonal(initial_link[1,:], 0)

                # calculate the updated link iteratively as in paper
                # to check the correcteness of the vectorized implementation
                predicted = np.zeros((2,4,4), dtype=np.float32)
                for i in range(4):
                    for j in range(4):
                        if i != j:
                            reset_factor = (1 - _write_weighting[:,i] - _write_weighting[:,j])
                            predicted[:, i, j]  = reset_factor * initial_link[:, i , j] + _write_weighting[:, i] * _precedence_vector[:, j]

                link_matrix = tf.convert_to_tensor(initial_link)
                precedence_vector = tf.convert_to_tensor(_precedence_vector)

                write_weighting = tf.constant(_write_weighting)

                op = mem.update_link_matrix(precedence_vector, link_matrix, write_weighting)
                L = session.run(op)

                self.assertTrue(np.allclose(L, predicted))


    def test_get_directional_weightings(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as session:

                mem = Memory(4, 5, 2, 2)
                _link_matrix = np.random.uniform(0, 1, (2, 4, 4)).astype(np.float32)
                _read_weightings = random_softmax((2, 4, 2), axis=1)
                predicted_forward = np.matmul(_link_matrix, _read_weightings)
                predicted_backward = np.matmul(np.transpose(_link_matrix, [0, 2, 1]), _read_weightings)

                read_weightings = tf.convert_to_tensor(_read_weightings)

                fop, bop = mem.get_directional_weightings(read_weightings, _link_matrix)

                forward_weighting, backward_weighting = session.run([fop, bop])

                self.assertTrue(np.allclose(forward_weighting, predicted_forward))
                self.assertTrue(np.allclose(backward_weighting, predicted_backward))



    def test_update_read_weightings(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as session:

                mem = Memory(4, 5, 2, 2)
                lookup_weightings = random_softmax((2, 4, 2), axis=1)
                forward_weighting = random_softmax((2, 4, 2), axis=1)
                backward_weighting = random_softmax((2, 4, 2), axis=1)
                read_mode = random_softmax((2, 3, 2), axis=1)
                predicted_weights = np.zeros((2, 4, 2)).astype(np.float32)

                # calculate the predicted weights using iterative method from paper
                # to check the correcteness of the vectorized implementation
                for i in range(2):
                    predicted_weights[:, :, i] = read_mode[:, 0,i, np.newaxis] * backward_weighting[:, :, i] + read_mode[:, 1, i, np.newaxis] * lookup_weightings[:, :, i] + read_mode[:, 2, i, np.newaxis] * forward_weighting[:, :, i]

                op = mem.update_read_weightings(lookup_weightings, forward_weighting, backward_weighting, read_mode)
                session.run(tf.initialize_all_variables())
                w_r = session.run(op)
                #updated_read_weightings = session.run(mem.read_weightings.value())

                self.assertTrue(np.allclose(w_r, predicted_weights))
                #self.assertTrue(np.allclose(updated_read_weightings, predicted_weights))


    def test_update_read_vectors(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph = graph) as session:

                mem = Memory(4, 5, 2, 4)
                memory_matrix = np.random.uniform(-1, 1, (4, 4, 5)).astype(np.float32)
                read_weightings = random_softmax((4, 4, 2), axis=1)
                predicted = np.matmul(np.transpose(memory_matrix, [0, 2, 1]), read_weightings)

                op = mem.update_read_vectors(memory_matrix, read_weightings)
                session.run(tf.initialize_all_variables())
                r = session.run(op)
                #updated_read_vectors = session.run(mem.read_vectors.value())

                self.assertTrue(np.allclose(r, predicted))
                #self.assertTrue(np.allclose(updated_read_vectors, predicted))

    def test_write(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph = graph) as session:

                mem = Memory(4, 5, 2, 1)
                M, u, p, L, ww, rw, r = session.run(mem.init_memory())
                key = np.random.uniform(0, 1, (1, 5, 1)).astype(np.float32)
                strength = np.random.uniform(0, 1, (1, 1)).astype(np.float32)
                free_gates = np.random.uniform(0, 1, (1, 2)).astype(np.float32)
                write_gate = np.random.uniform(0, 1, (1, 1)).astype(np.float32)
                allocation_gate = np.random.uniform(0, 1, (1,1)).astype(np.float32)
                write_vector = np.random.uniform(0, 1, (1, 5)).astype(np.float32)
                erase_vector = np.zeros((1, 5)).astype(np.float32)

                u_op, ww_op, M_op, L_op, p_op = mem.write(
                    M, u, rw, ww, p, L,
                    key, strength, free_gates, allocation_gate,
                    write_gate , write_vector, erase_vector
                )
                session.run(tf.initialize_all_variables())
                u, ww, M, L, p = session.run([u_op, ww_op, M_op, L_op, p_op])

                self.assertEqual(u.shape, (1, 4))
                self.assertEqual(ww.shape, (1, 4))
                self.assertEqual(M.shape, (1, 4, 5))
                self.assertEqual(L.shape, (1, 4, 4))
                self.assertEqual(p.shape, (1, 4))



    def test_read(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph = graph) as session:
                mem = Memory(4, 5, 2, 1)
                M, u, p, L, ww, rw, r = session.run(mem.init_memory())
                keys = np.random.uniform(0, 1, (1, 5, 2)).astype(np.float32)
                strengths = np.random.uniform(0, 1, (1, 2)).astype(np.float32)
                link_matrix = np.random.uniform(0, 1, (1, 4, 4)).astype(np.float32)
                read_modes = random_softmax((1, 3, 2), axis=1).astype(np.float32)
                memory_matrix = np.random.uniform(-1, 1, (1, 4, 5)).astype(np.float32)

                wr_op, r_op = mem.read(memory_matrix, rw, keys, strengths, link_matrix, read_modes)
                session.run(tf.initialize_all_variables())
                wr, r = session.run([wr_op, r_op])

                self.assertEqual(wr.shape, (1, 4, 2))
                self.assertEqual(r.shape, (1, 5, 2))


if __name__ == '__main__':
    unittest.main(verbosity=2)
