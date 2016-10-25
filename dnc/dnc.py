import tensorflow as tf
from memory import Memory

class DNC:

    def __init__(self, controller_class, input_size, output_size, memory_words_num = 256,
                 memory_word_size = 64, memory_read_heads = 4, batch_size = 1):
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
        self.words_num = memory_words_num
        self.word_size = memory_word_size
        self.read_heads = memory_read_heads
        self.batch_size = batch_size

        self.memory = Memory(self.words_num, self.word_size, self.read_heads, self.batch_size)
        self.controller = controller_class(self.input_size, self.output_size, self.read_heads, self.word_size)


    def __call__(self, X):
        """
        performs a step-by-step evaluation of the input data batches

        Parameters:
        ----------
        X: Tensor (batch_size, time_steps, input_size)
            the input data batch

        Returns: Tuple
            Tensor (batch_size, time_steps, output_size)
                the batch of outputs
            dict
                a view of memory gates and weightings for each time step
                (for visualization purposes)
        """

        time_steps = tf.unpack(X, axis=1)

        outputs = []
        free_gates = []
        allocation_gates = []
        write_gates = []
        read_weightings = []
        write_weightings = []

        for step in time_steps:

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

            output = self.controller.final_output(pre_output, new_read_vectors)
            outputs.append(output)

            # collecting into-memory view for the current step
            free_gates.append(interface['free_gates'])
            allocation_gates.append(interface['allocation_gate'])
            write_gates.append(interface['write_gate'])
            read_weightings.append(self.memory.read_weightings)
            write_weightings.append(self.memory.write_weighting)

        packed_output = tf.pack(outputs, axis=1)
        packed_memory_view = {
            'free_gates': tf.pack(free_gates, axis=1),
            'allocation_gates': tf.pack(allocation_gates, axis=1),
            'write_gates': tf.pack(write_gates, axis=1),
            'read_weightings': tf.pack(read_weightings, axis=1),
            'write_weightings': tf.pack(write_weightings, axis=1)
        }

        return packed_output, packed_memory_view
