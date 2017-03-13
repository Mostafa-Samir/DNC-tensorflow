import tensorflow as tf
import numpy as np

class BaseController:

    def __init__(self, input_size, output_size, memory_read_heads, memory_word_size, batch_size=1):
        """
        constructs a controller as described in the DNC paper:
        http://www.nature.com/nature/journal/vaop/ncurrent/full/nature20101.html

        Parameters:
        ----------
        input_size: int
            the size of the data input vector
        output_size: int
            the size of the data output vector
        memory_read_heads: int
            the number of read haeds in the associated external memory
        memory_word_size: int
            the size of the word in the associated external memory
        batch_size: int
            the size of the input data batch [optional, usually set by the DNC object]
        """

        self.input_size = input_size
        self.output_size = output_size
        self.read_heads = memory_read_heads
        self.word_size = memory_word_size
        self.batch_size = batch_size

        # indicates if the internal neural network is recurrent
        # by the existence of recurrent_update and get_state methods
        has_recurrent_update = callable(getattr(self, 'update_state', None))
        has_get_state = callable(getattr(self, 'get_state', None))
        self.has_recurrent_nn =  has_recurrent_update and has_get_state

        # the actual size of the neural network input after flatenning and
        # concatenating the input vector with the previously read vctors from memory
        self.nn_input_size = self.word_size * self.read_heads + self.input_size

        self.interface_vector_size = self.word_size * self.read_heads + 3 * self.word_size + 5 * self.read_heads + 3

        # define network vars
        with tf.name_scope("controller"):
            self.network_vars()

            self.nn_output_size = None
            with tf.variable_scope("shape_inference"):
                self.nn_output_size = self.get_nn_output_size()

            self.initials()

    def initials(self):
        """
        sets the initial values of the controller transformation weights matrices
        this method can be overwritten to use a different initialization scheme
        """
        # defining internal weights of the controller
        self.interface_weights = tf.Variable(
            tf.random_normal([self.nn_output_size, self.interface_vector_size], stddev=0.1),
            name='interface_weights'
        )
        self.nn_output_weights = tf.Variable(
            tf.random_normal([self.nn_output_size, self.output_size], stddev=0.1),
            name='nn_output_weights'
        )
        self.mem_output_weights = tf.Variable(
            tf.random_normal([self.word_size * self.read_heads, self.output_size],  stddev=0.1),
            name='mem_output_weights'
        )

    def network_vars(self):
        """
        defines the variables needed by the internal neural network
        [the variables should be attributes of the class, i.e. self.*]
        """
        raise NotImplementedError("network_vars is not implemented")


    def network_op(self, X):
        """
        defines the controller's internal neural network operation

        Parameters:
        ----------
        X: Tensor (batch_size, word_size * read_haeds + input_size)
            the input data concatenated with the previously read vectors from memory

        Returns: Tensor (batch_size, nn_output_size)
        """
        raise NotImplementedError("network_op method is not implemented")


    def get_nn_output_size(self):
        """
        retrives the output size of the defined neural network

        Returns: int
            the output's size

        Raises: ValueError
        """

        input_vector =  np.zeros([self.batch_size, self.nn_input_size], dtype=np.float32)
        output_vector = None

        if self.has_recurrent_nn:
            output_vector,_ = self.network_op(input_vector, self.get_state())
        else:
            output_vector = self.network_op(input_vector)

        shape = output_vector.get_shape().as_list()

        if len(shape) > 2:
            raise ValueError("Expected the neural network to output a 1D vector, but got %dD" % (len(shape) - 1))
        else:
            return shape[1]


    def parse_interface_vector(self, interface_vector):
        """
        pasres the flat interface_vector into its various components with their
        correct shapes

        Parameters:
        ----------
        interface_vector: Tensor (batch_size, interface_vector_size)
            the flattened inetrface vector to be parsed

        Returns: dict
            a dictionary with the components of the interface_vector parsed
        """

        parsed = {}

        r_keys_end = self.word_size * self.read_heads
        r_strengths_end = r_keys_end + self.read_heads
        w_key_end = r_strengths_end + self.word_size
        erase_end = w_key_end + 1 + self.word_size
        write_end = erase_end + self.word_size
        free_end = write_end + self.read_heads

        r_keys_shape = (-1, self.word_size, self.read_heads)
        r_strengths_shape = (-1, self.read_heads)
        w_key_shape = (-1, self.word_size, 1)
        write_shape = erase_shape = (-1, self.word_size)
        free_shape = (-1, self.read_heads)
        modes_shape = (-1, 3, self.read_heads)

        # parsing the vector into its individual components
        parsed['read_keys'] = tf.reshape(interface_vector[:, :r_keys_end], r_keys_shape)
        parsed['read_strengths'] = tf.reshape(interface_vector[:, r_keys_end:r_strengths_end], r_strengths_shape)
        parsed['write_key'] = tf.reshape(interface_vector[:, r_strengths_end:w_key_end], w_key_shape)
        parsed['write_strength'] = tf.reshape(interface_vector[:, w_key_end], (-1, 1))
        parsed['erase_vector'] = tf.reshape(interface_vector[:, w_key_end + 1:erase_end], erase_shape)
        parsed['write_vector'] = tf.reshape(interface_vector[:, erase_end:write_end], write_shape)
        parsed['free_gates'] = tf.reshape(interface_vector[:, write_end:free_end], free_shape)
        parsed['allocation_gate'] = tf.expand_dims(interface_vector[:, free_end], 1)
        parsed['write_gate'] = tf.expand_dims(interface_vector[:, free_end + 1], 1)
        parsed['read_modes'] = tf.reshape(interface_vector[:, free_end + 2:], modes_shape)

        # transforming the components to ensure they're in the right ranges
        parsed['read_strengths'] = 1 + tf.nn.softplus(parsed['read_strengths'])
        parsed['write_strength'] = 1 + tf.nn.softplus(parsed['write_strength'])
        parsed['erase_vector'] = tf.nn.sigmoid(parsed['erase_vector'])
        parsed['free_gates'] = tf.nn.sigmoid(parsed['free_gates'])
        parsed['allocation_gate'] = tf.nn.sigmoid(parsed['allocation_gate'])
        parsed['write_gate'] = tf.nn.sigmoid(parsed['write_gate'])
        parsed['read_modes'] = tf.nn.softmax(parsed['read_modes'], 1)

        return parsed

    def process_input(self, X, last_read_vectors, state=None):
        """
        processes input data through the controller network and returns the
        pre-output and interface_vector

        Parameters:
        ----------
        X: Tensor (batch_size, input_size)
            the input data batch
        last_read_vectors: (batch_size, word_size, read_heads)
            the last batch of read vectors from memory
        state: Tuple
            state vectors if the network is recurrent

        Returns: Tuple
            pre-output: Tensor (batch_size, output_size)
            parsed_interface_vector: dict
        """

        flat_read_vectors = tf.reshape(last_read_vectors, (-1, self.word_size * self.read_heads))
        complete_input = tf.concat([X, flat_read_vectors], 1)
        nn_output, nn_state = None, None

        if self.has_recurrent_nn:
            nn_output, nn_state = self.network_op(complete_input, state)
        else:
            nn_output = self.network_op(complete_input)

        pre_output = tf.matmul(nn_output, self.nn_output_weights)
        interface = tf.matmul(nn_output, self.interface_weights)
        parsed_interface = self.parse_interface_vector(interface)

        if self.has_recurrent_nn:
            return pre_output, parsed_interface, nn_state
        else:
            return pre_output, parsed_interface


    def final_output(self, pre_output, new_read_vectors):
        """
        returns the final output by taking rececnt memory changes into account

        Parameters:
        ----------
        pre_output: Tensor (batch_size, output_size)
            the ouput vector from the input processing step
        new_read_vectors: Tensor (batch_size, words_size, read_heads)
            the newly read vectors from the updated memory

        Returns: Tensor (batch_size, output_size)
        """

        flat_read_vectors = tf.reshape(new_read_vectors, (-1, self.word_size * self.read_heads))

        final_output = pre_output + tf.matmul(flat_read_vectors, self.mem_output_weights)

        return final_output
