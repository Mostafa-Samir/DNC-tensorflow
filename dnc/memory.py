import tensorflow as tf
import numpy as np
import utility

class Memory:

    def __init__(self, words_num=256, word_size=64, read_heads=4, batch_size=1):
        """
        constructs a memory matrix with read heads and a write head as described
        in the DNC paper
        http://www.nature.com/nature/journal/vaop/ncurrent/full/nature20101.html

        Parameters:
        ----------
        words_num: int
            the maximum number of words that can be stored in the memory at the
            same time
        word_size: int
            the size of the individual word in the memory
        read_heads: int
            the number of read heads that can read simultaneously from the memory
        batch_size: int
            the size of input data batch
        """

        self.words_num = words_num
        self.word_size = word_size
        self.read_heads = read_heads
        self.batch_size = batch_size

        with tf.name_scope('external_memory'):

            self.memory_matrix = tf.zeros([batch_size, words_num, word_size], name='memory_matrix')
            self.usage_vector = tf.zeros([batch_size, words_num, ], name='usage_vector')
            self.precedence_vector = tf.zeros([batch_size, words_num, ], name='precedence_vector')
            self.link_matrix = tf.zeros([batch_size, words_num, words_num], name='link_matrix')

            self.write_weighting = tf.zeros([batch_size, words_num, ], name='write_weighting')
            self.read_weightings = tf.zeros([batch_size, words_num, read_heads], name='read_weightings')

            self.read_vectors = tf.zeros([batch_size, word_size, read_heads])

            # a words_num x words_num identity matrix
            self.I = tf.constant(np.identity(words_num, dtype=np.float32))

            # a variable used to bring the allocation weighting in order of memory words
            # after computing them using the sorted usage and free list
            self.ordered_allocation_weighting = tf.Variable(tf.zeros([batch_size * words_num]), trainable=False)

            # maps the indecies from the 2D array of free list per batch to
            # their corresponding values in the flat 1D array of ordered_allocation_weighting
            self.index_mapper = tf.constant(np.cumsum([0] + [words_num] * (batch_size - 1), dtype=np.int32)[:, np.newaxis])



    def get_lookup_weighting(self, keys, strengths):
        """
        retrives a content-based adderssing weighting given the keys

        Parameters:
        ----------
        keys: Tensor (batch_size, word_size, number_of_keys)
            the keys to query the memory with
        strengths: Tensor (batch_size, number_of_keys, )
            the list of strengths for each lookup key

        Returns: Tensor (batch_size, words_num, number_of_keys)
            The list of lookup weightings for each provided key
        """

        normalized_memory = tf.nn.l2_normalize(self.memory_matrix, 2)
        normalized_keys = tf.nn.l2_normalize(keys, 1)

        similiarity = tf.batch_matmul(normalized_memory, normalized_keys)
        strengths = tf.expand_dims(strengths, 1)

        return tf.nn.softmax(similiarity * strengths, 1)


    def update_usage_vector(self, free_gates):
        """
        updates and returns the usgae vector given the values of the free gates

        Parameters:
        ----------
        free_gates: Tensor (batch_size, read_heads, )

        Returns: Tensor (batch_size, words_num, )
            the updated usage vector
        """
        free_gates = tf.expand_dims(free_gates, 1)

        retention_vector = tf.reduce_prod(1 - self.read_weightings * free_gates, 2)
        updated_usage = (self.usage_vector + self.write_weighting - self.usage_vector * self.write_weighting)  * retention_vector
        updated_usage = self.usage_vector.assign(updated_usage)

        return updated_usage


    def get_allocation_weighting(self, sorted_usage, free_list):
        """
        retreives the writing allocation weighting based on the usage free list

        Parameters:
        ----------
        sorted_usage: Tensor (batch_size, words_num, )
            the usage vector sorted ascndingly
        free_list: Tensor (batch, words_num, )
            the original indecies of the sorted usage vector

        Returns: Tensor (batch_size, words_num, )
            the allocation weighting for each word in memory
        """

        shifted_cumprod = tf.cumprod(sorted_usage, axis = 1, exclusive=True)
        unordered_allocation_weighting = (1 - sorted_usage) * shifted_cumprod

        mapped_free_list = free_list + self.index_mapper

        allocation_weighting = tf.scatter_update(self.ordered_allocation_weighting, mapped_free_list, unordered_allocation_weighting)
        allocation_weighting = tf.reshape(allocation_weighting, (self.batch_size, self.words_num))

        return allocation_weighting


    def update_write_weighting(self, lookup_weighting, allocation_weighting, write_gate, allocation_gate):
        """
        updates and returns the current write_weighting

        Parameters:
        ----------
        lookup_weighting: Tensor (batch_size, words_num, 1)
            the weight of the lookup operation in writing
        allocation_weighting: Tensor (batch_size, words_num)
            the weight of the allocation operation in writing
        write_gate: (batch_size, 1)
            the fraction of writing to be done
        allocation_gate: (batch_size, 1)
            the fraction of allocation to be done

        Returns: Tensor (batch_size, words_num)
            the updated write_weighting
        """

        # remove the dimension of 1 from the lookup_weighting
        lookup_weighting = tf.squeeze(lookup_weighting)

        updated_write_weighting = write_gate * (allocation_gate * allocation_weighting + (1 - allocation_gate) * lookup_weighting)
        updated_write_weighting = self.write_weighting.assign(updated_write_weighting)

        return updated_write_weighting


    def update_memory(self, write_weighting, write_vector, erase_vector):
        """
        updates and returns the memory matrix given the weighting and write and erase vectors

        Parameters:
        ----------
        write_weighting: Tensor (batch_size, words_num)
            the weight of writing at each memory location
        write_vector: Tensor (batch_size, word_size)
            a vector specifying what to write
        erase_vector: Tensor (batch_size, word_size)
            a vector specifying what to erase from memory

        Returns: Tensor (batch_size, words_num, word_size)
            the updated memory matrix
        """

        # expand data with a dimension of 1 at multiplication-adjacent location
        # to force matmul to behave as an outer product
        write_weighting = tf.expand_dims(write_weighting, 2)
        write_vector = tf.expand_dims(write_vector, 1)
        erase_vector = tf.expand_dims(erase_vector, 1)

        erasing = self.memory_matrix * (1 - tf.batch_matmul(write_weighting, erase_vector))
        writing = tf.batch_matmul(write_weighting, write_vector)
        updated_memory = self.memory_matrix.assign(erasing + writing)

        return updated_memory


    def update_precedence_vector(self, write_weighting):
        """
        updates the precedence vector given the latest write weighting

        Parameters:
        ----------
        write_weighting: Tensor (batch_size,words_num)
            the latest write weighting for the memory

        Returns: Tensor (batch_size, words_num)
            the updated precedence vector
        """

        reset_factor = 1 - tf.reduce_sum(write_weighting, 1, keep_dims=True)
        updated_precedence_vector = reset_factor * self.precedence_vector + write_weighting
        updated_precedence_vector = self.precedence_vector.assign(updated_precedence_vector)

        return updated_precedence_vector


    def update_link_matrix(self, write_weighting):
        """
        updates and returns the temporal link matrix gievn for the latest write

        Parameters:
        ----------
        write_weighting: Tensor (batch_size, words_num)
            the latest write_weighting for the memorye

        Returns: Tensor (batch_size, words_num, words_num)
            the updated temporal link matrix
        """

        write_weighting = tf.expand_dims(write_weighting, 2)
        precedence_vector = tf.expand_dims(self.precedence_vector, 1)

        reset_factor = 1 - utility.pairwise_add(write_weighting, is_batch=True)
        updated_link_matrix = reset_factor * self.link_matrix + tf.batch_matmul(write_weighting, precedence_vector)
        updated_link_matrix = (1 - self.I) * updated_link_matrix  # eliminates self-links
        updated_link_matrix = self.link_matrix.assign(updated_link_matrix)

        return updated_link_matrix


    def get_directional_weightings(self, link_matrix):
        """
        computes and returns the forward and backward reading weightings

        Parameters:
        ----------
        link_matrix: Tensor (batch_size, words_num, words_num)
            the temporal link matrix

        Returns: Tuple
            forward weighting: Tensor (batch_size, words_num, read_heads),
            backward weighting: Tensor (batch_size, words_num, read_heads)
        """

        forward_weighting = tf.batch_matmul(link_matrix, self.read_weightings)
        backward_weighting = tf.batch_matmul(link_matrix, self.read_weightings, adj_x=True)

        return forward_weighting, backward_weighting


    def update_read_weightings(self, lookup_weightings, forward_weighting, backward_weighting, read_mode):
        """
        updates and returns the current read_weightings

        Parameters:
        ----------
        lookup_weightings: Tensor (batch_size, words_num, read_heads)
            the content-based read weighting
        forward_weighting: Tensor (batch_size, words_num, read_heads)
            the forward direction read weighting
        backward_weighting: Tensor (batch_size, words_num, read_heads)
            the backward direction read weighting
        read_mode: Tesnor (batch_size, 3, read_heads)
            the softmax distribution between the three read modes

        Returns: Tensor (batch_size, words_num, read_heads)
        """

        backward_mode = tf.expand_dims(read_mode[:, 0, :], 1) * backward_weighting
        lookup_mode = tf.expand_dims(read_mode[:, 1, :], 1) * lookup_weightings
        forward_mode = tf.expand_dims(read_mode[:, 2, :], 1) * forward_weighting

        updated_read_weightings = self.read_weightings.assign(backward_mode + lookup_mode + forward_mode)

        return updated_read_weightings


    def update_read_vectors(self, memory_matrix, read_weightings):
        """
        reads, updates, and returns the read vectors of the recently updated memory

        Parameters:
        ----------
        memory_matrix: Tensor (batch_size, words_num, word_size)
            the recently updated memory matrix
        read_weightings: Tensor (batch_size, words_num, read_heads)
            the amount of info to read from each memory location by each read head

        Returns: Tensor (word_size, read_heads)
        """

        updated_read_vectors = tf.batch_matmul(memory_matrix, read_weightings, adj_x=True)
        updated_read_vectors = self.read_vectors.assign(updated_read_vectors)

        return updated_read_vectors


    def write(self, key, strength, free_gates, allocation_gate, write_gate, write_vector, erase_vector):
        """
        defines the complete pipeline of writing to memory gievn the write variables

        Parameters:
        ----------
        key: Tensor (batch_size, word_size, 1)
            the key to query the memory location with
        strength: (batch_size, 1)
            the strength of the query key
        free_gates: Tensor (batch_size, read_heads)
            the degree to which location at read haeds will be freed
        allocation_gate: (batch_size, 1)
            the fraction of writing that is being allocated in a new locatio
        write_gate: (batch_size, 1)
            the amount of information to be written to memory
        write_vector: Tensor (batch_size, word_size)
            specifications of what to write to memory
        erase_vector: Tensor(batch_size, word_size)
            specifications of what to erase from memory

        Returns : Tuple
            the updated memory_matrix: Tensor (batch_size, words_num, words_size)
            the updated link matrix: Tensor(batch_size, words_num, words_num)
        """

        lookup_weighting = self.get_lookup_weighting(key, strength)
        usage_vector = self.update_usage_vector(free_gates)

        sorted_usage, free_list = tf.nn.top_k(-1 * usage_vector, self.words_num)
        sorted_usage = -1 * sorted_usage

        allocation_weighting = self.get_allocation_weighting(sorted_usage, free_list)
        write_weighting = self.update_write_weighting(lookup_weighting, allocation_weighting, write_gate, allocation_gate)
        memory_matrix = self.update_memory(write_weighting, write_vector, erase_vector)
        link_matrix = self.update_link_matrix(write_weighting)
        self.update_precedence_vector(write_weighting)

        return memory_matrix, link_matrix


    def read(self, keys, strengths, link_matrix, read_modes, memory_matrix):
        """
        defines the complete pipeline for reading from memory

        Parameters:
        ----------
        keys: Tensor (batch_size, word_size, read_heads)
            the kyes to query the memory locations with
        strengths: Tensor (batch_size, read_heads)
            the strength of each read key
        link_matrix: Tensor (batch_size, words_num, words_num)
            the updated link matrix from the last writing
        read_modes: Tensor (batch_size, 3, read_heads)
            the softmax distribution between the three read modes
        memory_matrix: Tensor (batch_size, words_num, word_size)
            the updated memory matrix from the last writing

        Returns: Tensor (batch_size, word_size, read_heads)
            the recently read vectors
        """

        lookup_weighting = self.get_lookup_weighting(keys, strengths)
        forward_weighting, backward_weighting = self.get_directional_weightings(link_matrix)
        read_weightings = self.update_read_weightings(lookup_weighting, forward_weighting, backward_weighting, read_modes)
        read_vectors = self.update_read_vectors(memory_matrix, read_weightings)

        return read_vectors
