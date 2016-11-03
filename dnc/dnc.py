import tensorflow as tf
from memory import Memory
import os

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
        self.controller = controller_class(self.input_size, self.output_size, self.read_heads, self.word_size, self.batch_size)

        # input data placeholders
        self.input_data = tf.placeholder(tf.float32, [batch_size, None, input_size], name='input')
        self.target_output = tf.placeholder(tf.float32, [batch_size, None, output_size], name='targets')
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
        pre_output, interface, nn_state = None, None, None

        if self.controller.has_recurrent_nn:
            pre_output, interface, nn_state = self.controller.process_input(step, last_read_vectors)
        else:
            pre_output, interface = self.controller.process_input(step, last_read_vectors)

        usage_vector, write_weighting, memory_matrix, link_matrix, precedence_vector = self.memory.write(
            interface['write_key'],
            interface['write_strength'],
            interface['free_gates'],
            interface['allocation_gate'],
            interface['write_gate'],
            interface['write_vector'],
            interface['erase_vector']
        )

        read_weightings, read_vectors = self.memory.read(
            interface['read_keys'],
            interface['read_strengths'],
            link_matrix,
            interface['read_modes'],
            memory_matrix
        )

        return [

            # report new memory state to be updated outside the condition branch
            usage_vector,
            write_weighting,
            memory_matrix,
            link_matrix,
            precedence_vector,
            read_weightings,
            read_vectors,

            self.controller.final_output(pre_output, read_vectors),
            interface['free_gates'],
            interface['allocation_gate'],
            interface['write_gate'],

            # report new state of RNN if exists
            nn_state[0] if nn_state is not None else tf.zeros(1),
            nn_state[1] if nn_state is not None else tf.zeros(1)
        ]


    def _dummy_op(self):
        """
        returns dummy outputs for padded, out of sequence inputs
        """

        nn_state = None
        if self.controller.has_recurrent_nn:
            # get the current RNN state unchanged
            nn_state = self.controller.get_state()

        return [

            # report the current memory state unchanged
            self.memory.usage_vector,
            self.memory.write_weighting,
            self.memory.memory_matrix,
            self.memory.link_matrix,
            self.memory.precedence_vector,
            self.memory.read_weightings,
            self.memory.read_vectors,

            tf.zeros([self.batch_size, self.output_size]),
            tf.zeros([self.batch_size, self.read_heads]),
            tf.zeros([self.batch_size, 1]),
            tf.zeros([self.batch_size, 1]),

            # report state of RNN if exists
            nn_state[0] if nn_state is not None else tf.zeros(1),
            nn_state[1] if nn_state is not None else tf.zeros(1)
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
        dependencies = [tf.no_op()]

        with tf.variable_scope("sequence_loop") as scope:
            for t, step in enumerate(time_steps):

                with tf.control_dependencies(dependencies):
                    output_list = tf.cond(t < self.sequence_length,
                        # if step is within the sequence_length, perform regualr operations
                        lambda: self._step_op(step),
                        # otherwise: perform dummy operation
                        self._dummy_op
                    )

                    scope.reuse_variables()

                    dependencies = [
                        self.memory.usage_vector.assign(output_list[0]),
                        self.memory.write_weighting.assign(output_list[1]),
                        self.memory.memory_matrix.assign(output_list[2]),
                        self.memory.link_matrix.assign(output_list[3]),
                        self.memory.precedence_vector.assign(output_list[4]),
                        self.memory.read_weightings.assign(output_list[5]),
                        self.memory.read_vectors.assign(output_list[6]),
                    ]

                    if self.controller.has_recurrent_nn:
                        new_nn_state = (output_list[11], output_list[12])
                        dependencies.append(
                            self.controller.recurrent_update(new_nn_state)
                        )

                    outputs.append(output_list[7])

                    # collecting memory view for the current step
                    free_gates.append(output_list[8])
                    allocation_gates.append(output_list[9])
                    write_gates.append(output_list[10])
                    read_weightings.append(output_list[5])
                    write_weightings.append(output_list[1])

        with tf.control_dependencies(dependencies):
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


    def save(self, session, ckpts_dir, name):
        """
        saves the current values of the model's parameters to a checkpoint

        Parameters:
        ----------
        session: tf.Session
            the tensorflow session to save
        ckpts_dir: string
            the path to the checkpoints directories
        name: string
            the name of the checkpoint subdirectory
        """
        checkpoint_dir = os.path.join(ckpts_dir, name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        tf.train.Saver(tf.trainable_variables()).save(session, os.path.join(checkpoint_dir, 'model.ckpt'))


    def restore(self, session, ckpts_dir, name):
        """
        session: tf.Session
            the tensorflow session to restore into
        ckpts_dir: string
            the path to the checkpoints directories
        name: string
            the name of the checkpoint subdirectory
        """
        tf.train.Saver(tf.trainable_variables()).restore(session, os.path.join(ckpts_dir, name, 'model.ckpt'))
