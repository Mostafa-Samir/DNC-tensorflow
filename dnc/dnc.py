import tensorflow as tf
from memory import Memory

class DNC:

    def __init__(self, controller_class, input_size, output_size, max_sequence_length,
                 memory_words_num = 256, memory_word_size = 64, memory_read_heads = 4, batch_size = 1):
        """
        constructs a complete DNC architecture as described in the DNC paper
        http://www.nature.com/nature/journal/vaop/ncurrent/full/nature20101.html

        Parameters:
        -----------
        controller_class: BaseController
            a concrete implementation of the BaseController class
        input_size: int
            the size of the input vector
        output_size: int
            the size of the output vector
        max_sequence_length: int
            the maximum length of an input sequence
        memory_words_num: int
            the number of words that can be stored in memory
        memory_word_size: int
            the size of an individual word in memory
        memory_read_heads: int
            the number of read heads in the memory
        batch_size: int
            the size of the data batch
        """

        self.input_size = input_size
        self.output_size = output_size
        self.max_sequence_length = max_sequence_length
        self.words_num = memory_words_num
        self.word_size = memory_word_size
        self.read_heads = memory_read_heads
        self.batch_size = batch_size

        self.memory = Memory(self.words_num, self.word_size, self.read_heads, self.batch_size)
        self.controller = controller_class(self.input_size, self.output_size, self.read_heads, self.word_size)

        # input data placeholders
        self.input_data = tf.placeholder(tf.float32, [batch_size, None, input_size], name='input')
        self.sequence_length = tf.placeholder(tf.int32, name='sequence_length')

        self.input_padding = tf.zeros([batch_size, max_sequence_length, input_size])

        self.build_graph()


    def _step_op(self, step):
        """
        performs a step operation on the input step data

        Parameters:
        ----------
        step: Tensor (batch_size, input_size)

        Returns: Tuple
            output: Tensor (batch_size, output_size)
            memory_view: dict
        """

        last_read_vectors = self.memory.read_vectors
        pre_output, interface = self.controller.process_input(step, last_read_vectors)

        memory_matrix, link_matrix = self.memory.write(
            interface['write_key'],
            interface['write_strength'],
            interface['free_gates'],
            interface['allocation_gate'],
            interface['write_gate'],
            interface['write_vector'],
            interface['erase_vector']
        )

        new_read_vectors = self.memory.read(
            interface['read_keys'],
            interface['read_strengths'],
            link_matrix,
            interface['read_modes'],
            memory_matrix
        )

        return [
            self.controller.final_output(pre_output, new_read_vectors),
            interface['free_gates'],
            interface['allocation_gate'],
            interface['write_gate'],
            self.memory.read_weightings,
            self.memory.write_weighting
        ]


    def _dummy_op(self):
        """
        returns dummy outputs for padded, out of sequence inputs
        """

        return [
            tf.zeros([self.batch_size, self.output_size]),
            tf.zeros([self.batch_size, self.read_heads]),
            tf.zeros([self.batch_size, 1]),
            tf.zeros([self.batch_size, 1]),
            tf.zeros([self.batch_size, self.words_num, self.read_heads]),
            tf.zeros([self.batch_size, self.words_num])
        ]


    def build_graph(self):
        """
        builds the computational graph that performs a step-by-step evaluation
        of the input data batches
        """

        padding = tf.slice(self.input_padding, [0, self.sequence_length, 0], [-1, -1, -1])
        data = tf.concat(1, [self.input_data, padding])

        time_steps = tf.unpack(data, num=self.max_sequence_length, axis=1)

        outputs = []
        free_gates = []
        allocation_gates = []
        write_gates = []
        read_weightings = []
        write_weightings = []

        for t, step in enumerate(time_steps):

            output_list = tf.cond(t < self.sequence_length,
                # if step is within the sequence_length, perform regualr operations
                lambda: self._step_op(step),
                # otherwise: perform dummy operation
                self._dummy_op
            )

            outputs.append(output_list[0])

            # collecting memory view for the current step
            free_gates.append(output_list[1])
            allocation_gates.append(output_list[2])
            write_gates.append(output_list[3])
            read_weightings.append(output_list[4])
            write_weightings.append(output_list[5])

        self.packed_output = tf.slice(tf.pack(outputs, axis=1), [0, 0, 0], [-1, self.sequence_length, -1])
        self.packed_memory_view = {
            'free_gates': tf.slice(tf.pack(free_gates, axis=1), [0, 0, 0], [-1, self.sequence_length, -1]),
            'allocation_gates': tf.slice(tf.pack(allocation_gates, axis=1), [0, 0, 0], [-1, self.sequence_length, -1]),
            'write_gates': tf.slice(tf.pack(write_gates, axis=1), [0, 0, 0], [-1, self.sequence_length, -1]),
            'read_weightings': tf.slice(tf.pack(read_weightings, axis=1), [0, 0, 0, 0], [-1, self.sequence_length, -1, -1]),
            'write_weightings': tf.slice(tf.pack(write_weightings, axis=1), [0, 0, 0], [-1, self.sequence_length, -1])
        }


    def get_outputs(self):
        """
        returns the graph nodes for the output and memory view

        Returns: Tuple
            outputs: Tensor (batch_size, time_steps, output_size)
            memory_view: dict
        """
        return self.packed_output, self.packed_memory_view
