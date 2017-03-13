import tensorflow as tf
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import LSTMStateTuple
from memory import Memory
import utility
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

        self.build_graph()


    def _step_op(self, step, memory_state, controller_state=None):
        """
        performs a step operation on the input step data

        Parameters:
        ----------
        step: Tensor (batch_size, input_size)
        memory_state: Tuple
            a tuple of current memory parameters
        controller_state: Tuple
            the state of the controller if it's recurrent

        Returns: Tuple
            output: Tensor (batch_size, output_size)
            memory_view: dict
        """

        last_read_vectors = memory_state[6]
        pre_output, interface, nn_state = None, None, None

        if self.controller.has_recurrent_nn:
            pre_output, interface, nn_state = self.controller.process_input(step, last_read_vectors, controller_state)
        else:
            pre_output, interface = self.controller.process_input(step, last_read_vectors)

        usage_vector, write_weighting, memory_matrix, link_matrix, precedence_vector = self.memory.write(
            memory_state[0], memory_state[1], memory_state[5],
            memory_state[4], memory_state[2], memory_state[3],
            interface['write_key'],
            interface['write_strength'],
            interface['free_gates'],
            interface['allocation_gate'],
            interface['write_gate'],
            interface['write_vector'],
            interface['erase_vector']
        )

        read_weightings, read_vectors = self.memory.read(
            memory_matrix,
            memory_state[5],
            interface['read_keys'],
            interface['read_strengths'],
            link_matrix,
            interface['read_modes'],
        )

        return [

            # report new memory state to be updated outside the condition branch
            memory_matrix,
            usage_vector,
            precedence_vector,
            link_matrix,
            write_weighting,
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


    def _loop_body(self, time, memory_state, outputs, free_gates, allocation_gates, write_gates,
                   read_weightings, write_weightings, usage_vectors, controller_state):
        """
        the body of the DNC sequence processing loop

        Parameters:
        ----------
        time: Tensor
        outputs: TensorArray
        memory_state: Tuple
        free_gates: TensorArray
        allocation_gates: TensorArray
        write_gates: TensorArray
        read_weightings: TensorArray,
        write_weightings: TensorArray,
        usage_vectors: TensorArray,
        controller_state: Tuple

        Returns: Tuple containing all updated arguments
        """

        step_input = self.unpacked_input_data.read(time)

        output_list = self._step_op(step_input, memory_state, controller_state)

        # update memory parameters

        new_controller_state = tf.zeros(1)
        new_memory_state = tuple(output_list[0:7])

        new_controller_state = LSTMStateTuple(output_list[11], output_list[12])

        outputs = outputs.write(time, output_list[7])

        # collecting memory view for the current step
        free_gates = free_gates.write(time, output_list[8])
        allocation_gates = allocation_gates.write(time, output_list[9])
        write_gates = write_gates.write(time, output_list[10])
        read_weightings = read_weightings.write(time, output_list[5])
        write_weightings = write_weightings.write(time, output_list[4])
        usage_vectors = usage_vectors.write(time, output_list[1])

        return (
            time + 1, new_memory_state, outputs,
            free_gates,allocation_gates, write_gates,
            read_weightings, write_weightings,
            usage_vectors, new_controller_state
        )


    def build_graph(self):
        """
        builds the computational graph that performs a step-by-step evaluation
        of the input data batches
        """

        self.unpacked_input_data = utility.unpack_into_tensorarray(self.input_data, 1, self.sequence_length)

        outputs = tf.TensorArray(tf.float32, self.sequence_length)
        free_gates = tf.TensorArray(tf.float32, self.sequence_length)
        allocation_gates = tf.TensorArray(tf.float32, self.sequence_length)
        write_gates = tf.TensorArray(tf.float32, self.sequence_length)
        read_weightings = tf.TensorArray(tf.float32, self.sequence_length)
        write_weightings = tf.TensorArray(tf.float32, self.sequence_length)
        usage_vectors = tf.TensorArray(tf.float32, self.sequence_length)

        controller_state = self.controller.get_state() if self.controller.has_recurrent_nn else (tf.zeros(1), tf.zeros(1))
        memory_state = self.memory.init_memory()
        if not isinstance(controller_state, LSTMStateTuple):
            controller_state = LSTMStateTuple(controller_state[0], controller_state[1])
        final_results = None

        with tf.variable_scope("sequence_loop") as scope:
            time = tf.constant(0, dtype=tf.int32)

            final_results = tf.while_loop(
                cond=lambda time, *_: time < self.sequence_length,
                body=self._loop_body,
                loop_vars=(
                    time, memory_state, outputs,
                    free_gates, allocation_gates, write_gates,
                    read_weightings, write_weightings,
                    usage_vectors, controller_state
                ),
                parallel_iterations=32,
                swap_memory=True
            )

        dependencies = []
        if self.controller.has_recurrent_nn:
            dependencies.append(self.controller.update_state(final_results[9]))

        with tf.control_dependencies(dependencies):
            self.packed_output = utility.pack_into_tensor(final_results[2], axis=1)
            self.packed_memory_view = {
                'free_gates': utility.pack_into_tensor(final_results[3], axis=1),
                'allocation_gates': utility.pack_into_tensor(final_results[4], axis=1),
                'write_gates': utility.pack_into_tensor(final_results[5], axis=1),
                'read_weightings': utility.pack_into_tensor(final_results[6], axis=1),
                'write_weightings': utility.pack_into_tensor(final_results[7], axis=1),
                'usage_vectors': utility.pack_into_tensor(final_results[8], axis=1)
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
