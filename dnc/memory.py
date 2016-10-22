import tensorflow as tf
import numpy as np
import utility

class Memory:

    def __init__(self, words_num=256, word_size=64, read_heads=4):
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
        """

        self.words_num = words_num
        self.word_size = word_size
        self.read_heads = read_heads

        with tf.variable_scope('external_memory'):

            self.memory_matrix = tf.Variable(tf.zeros([words_num, word_size]), name='memory_matrix', trainable=False)
            self.usage_vector = tf.Variable(tf.zeros([words_num, ]), name='usage_vector', trainable=False)
            self.precedence_vector = tf.Variable(tf.zeros([words_num, ]), name='precedence_vector', trainable=False)
            self.link_matrix = tf.Variable(tf.zeros([words_num, words_num]), name='link_matrix', trainable=False)

            self.write_weighting = tf.Variable(tf.zeros([words_num, ]), name='write_weighting', trainable=False)
            self.read_weightings = tf.Variable(tf.zeros([words_num, read_heads]), name='read_weightings', trainable=False)

            self.read_vectors = tf.Variable(tf.zeros([word_size, read_heads]), name='read_vectors', trainable=False)

            # a words_num x words_num identity matrix
            self.I = tf.constant(np.identity(words_num, dtype=np.float32))

            # a variable used to bring the allocation weighting in order of memory words
            # after computing them using the sorted usage and free list
            self.ordered_allocation_weighting = tf.Variable(tf.zeros([words_num, ]), trainable=False)



    def get_lookup_weighting(self, keys, strengths):
        """
        retrives a content-based adderssing weighting given the keys

        Parameters:
        ----------
        keys: Tensor (word_size, number_of_keys)
            the keys to query the memory with
        strengths: Tensor (number_of_keys, )
            the list of strengths for each lookup key

        Returns: Tensor (words_num, number_of_keys)
            The list of lookup weightings for each provided key
        """

        normalized_memory = tf.nn.l2_normalize(self.memory_matrix, 1)
        normalized_keys = tf.nn.l2_normalize(keys, 1)

        similiarity = tf.matmul(normalized_memory, normalized_keys)

        return tf.nn.softmax(similiarity * strengths, 0)


    def update_usage_vector(self, free_gates):
        """
        updates and returns the usgae vector given the values of the free gates

        Parameters:
        ----------
        free_gates: Tensor (read_heads, )

        Returns: Tensor (words_num, )
            the updated usage vector
        """

        retention_vector = tf.reduce_prod(1 - self.read_weightings * free_gates, 1)
        updated_usage = (self.usage_vector + self.write_weighting - self.usage_vector * self.write_weighting)  * retention_vector
        updated_usage = self.usage_vector.assign(updated_usage)

        return updated_usage


    def get_allocation_weighting(self, sorted_usage, free_list):
        """
        retreives the writing allocation weighting based on the usage free list

        Parameters:
        ----------
        sorted_usage: Tensor (words_num, )
            the usage vector sorted ascndingly
        free_list: Tensor (words_num, )
            the original indecies of the sorted usage vector

        Returns: Tensor (words_num, )
            the allocation weighting for each word in memory
        """

        shifted_cumprod = tf.cumprod(sorted_usage, exclusive=True)
        unordered_allocation_weighting = (1 - sorted_usage) * shifted_cumprod

        allocation_weighting = tf.scatter_update(self.ordered_allocation_weighting, free_list, unordered_allocation_weighting)

        return allocation_weighting


    def update_write_weighting(self, lookup_weighting, allocation_weighting, write_gate, allocation_gate):
        """
        updates and returns the current write_weighting

        Parameters:
        ----------
        lookup_weighting: Tensor (number_of_keys, words_num)
            the weight of the lookup operation in writing
        allocation_weighting: Tensor (words_num, )
            the weight of the allocation operation in writing
        write_gate: Scalar
            the fraction of writing to be done
        allocation_gate: Scalar
            the fraction of allocation to be done

        Returns: Tensor (words_num, )
            the updated write_weighting
        """

        # for writing the lookup_weighting will be of shape (1, words_num)
        # becuase there's only one write key to query the memory with
        # so it can be reshaped to 1D safely
        lookup_weighting = tf.reshape(lookup_weighting, [-1, ])

        updated_write_weighting = write_gate * (allocation_gate * allocation_weighting + (1 - allocation_gate) * lookup_weighting)
        updated_write_weighting = self.write_weighting.assign(updated_write_weighting)

        return updated_write_weighting


    def update_memory(self, write_weighting, write_vector, erase_vector):
        """
        updates and returns the memory matrix given the weighting and write and erase vectors

        Parameters:
        ----------
        write_weighting: Tensor (words_num, )
            the weight of writing at each memory location
        write_vector: Tensor (word_size, )
            a vector specifying what to write
        erase_vector: Tensor (word_size, )
            a vector specifying what to erase from memory

        Returns: Tensor (words_num, word_size)
            the updated memory matrix
        """

        # transform vectors to 2D arrays with the last dimension being 1
        # to be able to carry out the matmuls as outer products
        write_weighting = tf.reshape(write_weighting, [-1, 1])
        write_vector = tf.reshape(write_vector, [-1, 1])
        erase_vector = tf.reshape(erase_vector, [-1, 1])

        erasing = self.memory_matrix * (1 - tf.matmul(write_weighting, erase_vector, transpose_b=True))
        writing = tf.matmul(write_weighting, write_vector, transpose_b=True)
        updated_memory = self.memory_matrix.assign(erasing + writing)

        return updated_memory


    def update_precedence_vector(self, write_weighting):
        """
        updates the precedence vector given the latest write weighting

        Parameters:
        ----------
        write_weighting: Tensor (words_num, )
            the latest write weighting for the memory

        Returns: Tensor (words_num, )
            the updated precedence vector
        """

        reset_factor = 1 - tf.reduce_sum(write_weighting)
        updated_precedence_vector = reset_factor * self.precedence_vector + write_weighting
        updated_precedence_vector = self.precedence_vector.assign(updated_precedence_vector)

        return updated_precedence_vector


    def update_link_matrix(self, write_weighting):
        """
        updates and returns the temporal link matrix gievn for the latest write

        Parameters:
        ----------
        write_weighting: Tensor (words_num, )
            the latest write_weighting for the memorye

        Returns: Tensor (words_num, words_num)
            the updated temporal link matrix
        """

        write_weighting = tf.reshape(write_weighting, [-1, 1])
        precedence_vector = tf.reshape(self.precedence_vector, [-1, 1])

        reset_factor = 1 - utility.pairwise_add(write_weighting)
        updated_link_matrix = reset_factor * self.link_matrix + tf.matmul(write_weighting, precedence_vector, transpose_b=True)
        updated_link_matrix = (1 - self.I) * updated_link_matrix  # eliminates self-links
        updated_link_matrix = self.link_matrix.assign(updated_link_matrix)

        return updated_link_matrix


    def get_directional_weightings(self, link_matrix):
        """
        computes and returns the forward and backward reading weightings

        Parameters:
        ----------
        link_matrix: Tensor (words_num, words_num)
            the temporal link matrix

        Returns: Tuple
            forward weighting: Tensor (words_num, read_heads),
            backward weighting: Tensor (words_num, read_heads)
        """

        forward_weighting = tf.matmul(link_matrix, self.read_weightings)
        backward_weighting = tf.matmul(link_matrix, self.read_weightings, transpose_a=True)

        return forward_weighting, backward_weighting


    def update_read_weightings(self, lookup_weightings, forward_weighting, backward_weighting, read_mode):
        """
        updates and returns the current read_weightings

        Parameters:
        ----------
        lookup_weightings: Tensor (words_num, read_heads)
            the content-based read weighting
        forward_weighting: Tensor (words_num, read_heads)
            the forward direction read weighting
        backward_weighting: Tensor (words_num, read_heads)
            the backward direction read weighting
        read_mode: Tesnor (3, read_heads)
            the softmax distribution between the three read modes

        Returns: Tensor (words_num, read_heads)
        """

        backward_mode = read_mode[0, :] * backward_weighting
        lookup_mode = read_mode[1, :] * lookup_weightings
        forward_mode = read_mode[2, :] * forward_weighting

        updated_read_weightings = self.read_weightings.assign(backward_mode + lookup_mode + forward_mode)

        return updated_read_weightings


    def update_read_vectors(self, memory_matrix, read_weightings):
        """
        reads, updates, and returns the read vectors of the recently updated memory

        Parameters:
        ----------
        memory_matrix: Tensor (words_num, word_size)
            the recently updated memory matrix
        read_weightings: Tensor (words_num, read_heads)
            the amount of info to read from each memory location by each read head

        Returns: Tensor (word_size, read_heads)
        """

        updated_read_vectors = tf.matmul(memory_matrix, read_weightings, transpose_a=True)
        updated_read_vectors = self.read_vectors.assign(updated_read_vectors)

        return updated_read_vectors


    def write(self, key, strength, free_gates, allocation_gate, write_gate, write_vector, erase_vector):
        """
        defines the complete pipeline of writing to memory gievn the write variables

        Parameters:
        ----------
        key: Tensor (word_size, )
            the key to query the memory location with
        strength: (1, )
            the strength of the query key
        free_gates: Tensor (read_heads,)
            the degree to which location at read haeds will be freed
        allocation_gate: Scalar
            the fraction of writing that is being allocated in a new locatio
        write_gate: Scalar
            the amount of information to be written to memory
        write_vector: Tensor (word_size, )
            specifications of what to write to memory
        erase_vector: Tensor(word_size, )
            specifications of what to erase from memory

        Returns : Tuple
            the updated memory_matrix: Tensor (words_num, words_size)
            the updated link matrix: Tensor(words_num, words_num)
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
        keys: Tensor (word_size, read_heads)
            the kyes to query the memory locations with
        strengths: Tensor (read_heads, )
            the strength of each read key
        link_matrix: Tensor (words_num, words_num)
            the updated link matrix from the last writing
        read_modes: Tensor (3, read_heads)
            the softmax distribution between the three read modes
        memory_matrix: Tensor (words_num, word_size)
            the updated memory matrix from the last writing

        Returns: Tensor (word_size, read_heads)
            the recently read vectors
        """

        lookup_weighting = self.get_lookup_weighting(keys, strengths)
        forward_weighting, backward_weighting = self.get_directional_weightings(link_matrix)
        read_weightings = self.update_read_weightings(lookup_weighting, forward_weighting, backward_weighting, read_modes)
        read_vectors = self.update_read_vectors(memory_matrix, read_weightings)

        return read_vectors
