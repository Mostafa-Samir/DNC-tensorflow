from __future__ import division, print_function
import warnings
import time
from datetime import timedelta
warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np
import getopt
import sys
import os

from dnc.dnc import DNC
from dnc.memory_visualization import visualize_op
from feedforward_controller import FeedforwardController


def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()


def generate_data(batch_size, length, size):

    input_data = np.zeros((batch_size, 2 * length + 1, size), dtype=np.float32)
    target_output = np.zeros((batch_size, 2 * length + 1, size), dtype=np.float32)

    sequence = np.random.binomial(1, 0.5, (batch_size, length, size - 1))

    input_data[:, :length, :size - 1] = sequence
    input_data[:, length, -1] = 1  # the end symbol
    target_output[:, length + 1:, :size - 1] = sequence

    return input_data, target_output


def binary_cross_entropy(predictions, targets):

    return tf.reduce_mean(
        -1 * targets * tf.log(predictions) - (1 - targets) * tf.log(1 - predictions)
    )

if __name__ == '__main__':
    # no any training - just log graph
    LOG_GRAPH_WITHOUT_OPTIMIZER = False
    seq_repeat = 5

    dirname = os.path.dirname(__file__)
    ckpts_dir = os.path.join(dirname , 'checkpoints')

    max_input_size = 5
    length = 'random_length'
    # length = 'fixed_length'
    batch_size = 1
    input_size = output_size = 6
    sequence_max_length = 11 * seq_repeat
    words_count = 10  # cells quantity
    word_size = 5  # size of each cell
    read_heads = 1

    images_dir = os.path.join(
        dirname,
        'images_max_len_{}_cells_qtty_{}_cell_size_{}_{}'.format(
            sequence_max_length, words_count, word_size, length))
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    tb_logs_dir = os.path.join(
        dirname,
        'logs_max_len_{}_cells_qtty_{}_cell_size_{}_{}'.format(
            sequence_max_length, words_count, word_size, length))
    print("logs dir:\n", tb_logs_dir)

    learning_rate = 1e-4
    momentum = 0.9

    from_checkpoint = None
    iterations = 100000

    options,_ = getopt.getopt(sys.argv[1:], '', ['checkpoint=', 'iterations='])

    for opt in options:
        if opt[0] == '--checkpoint':
            from_checkpoint = opt[1]
        elif opt[0] == '--iterations':
            iterations = int(opt[1])

    graph = tf.Graph()

    with graph.as_default():
        with tf.Session(graph=graph) as session:

            llprint("Building Computational Graph ... ")

            ncomputer = DNC(
                FeedforwardController,
                input_size,
                output_size,
                sequence_max_length,
                words_count,
                word_size,
                read_heads,
                batch_size
            )

            if LOG_GRAPH_WITHOUT_OPTIMIZER:
                summerizer = tf.train.SummaryWriter(tb_logs_dir, session.graph)
                session.run(tf.initialize_all_variables())
                exit()

            # squash the DNC output between 0 and 1
            output, packed_memory_view = ncomputer.get_outputs()
            squashed_output = tf.clip_by_value(tf.sigmoid(output), 1e-6, 1. - 1e-6)

            loss = binary_cross_entropy(squashed_output, ncomputer.target_output)

            summeries = []

            optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=momentum)
            gradients = optimizer.compute_gradients(loss)
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    summeries.append(tf.histogram_summary(var.name + '/grad', grad))
                    gradients[i] = (tf.clip_by_value(grad, -10, 10), var)

            apply_gradients = optimizer.apply_gradients(gradients)

            summeries.append(tf.scalar_summary("Loss", loss))

            # summerize_op = tf.merge_summary(summeries)
            summerize_op = tf.merge_all_summaries()
            no_summerize = tf.no_op()

            summerizer = tf.train.SummaryWriter(tb_logs_dir, session.graph)

            llprint("Done!\n")

            llprint("Initializing Variables ... ")
            session.run(tf.initialize_all_variables())
            llprint("Done!\n")

            if from_checkpoint is not None:
                llprint("Restoring Checkpoint %s ... " % (from_checkpoint))
                ncomputer.restore(session, ckpts_dir, from_checkpoint)
                llprint("Done!\n")


            last_100_losses = []
            accumulated_acc = []
            start_time = time.time()
            for i in range(iterations + 1):
            # for i in range(3):
                llprint("\rIteration %d/%d" % (i, iterations))

                inputs, targets = [], []
                for _ in range(seq_repeat):
                    if length == 'random_length':
                        random_length = np.random.randint(1, max_input_size)
                    else:
                        random_length = max_input_size
                    input_data, target_output = generate_data(batch_size, random_length, input_size)
                    inputs.append(input_data)
                    targets.append(target_output)
                input_data = np.concatenate(inputs, axis=1)
                target_output = np.concatenate(targets, axis=1)

                summerize = (i % 500 == 0)
                # summerize = False
                take_checkpoint = (i != 0) and (i % iterations == 0)

                loss_value, _, summary, mem_view, out = session.run([
                    loss,
                    apply_gradients,
                    summerize_op if summerize else no_summerize,
                    packed_memory_view,
                    squashed_output,
                ], feed_dict={
                    ncomputer.input_data: input_data,
                    ncomputer.target_output: target_output,
                    ncomputer.sequence_length: input_data.shape[1]
                })
                out = np.around(out)
                acc = np.mean(out == target_output)
                # keys = sorted(list(mem_view.keys()))
                # for k in keys:
                #     print(k, '\n', mem_view[k].shape)

                last_100_losses.append(loss_value)
                accumulated_acc.append(acc)
                summerizer.add_summary(summary, i)

                if summerize:
                    visualize_op(input_data, target_output, mem_view, images_dir, i)
                    llprint("\n\tAvg. Logistic Loss: %.4f, accuracy: %.4f\n" % (np.mean(last_100_losses), np.mean(accumulated_acc)))
                    time_cons_per_iteration = (time.time() - start_time) / (i + 1)
                    print("ETC: ", str(timedelta(seconds=time_cons_per_iteration * (iterations - i))))
                    # print("out\n", out[:, -random_length:])
                    # print("target\n", target_output[:, -random_length:])
                    last_100_losses = []
                    accumulated_acc = []
                    # validation
                    validation_loss = []
                    validation_acc = []
                    for j in range(10):
                        if length == 'random_length':
                            random_length = np.random.randint(1, max_input_size)
                        else:
                            random_length = max_input_size
                        input_data, target_output = generate_data(batch_size, random_length, input_size)
                        loss_value, out = session.run([
                            loss,
                            squashed_output,
                        ], feed_dict={
                            ncomputer.input_data: input_data,
                            ncomputer.target_output: target_output,
                            ncomputer.sequence_length: input_data.shape[1]
                        })
                        out = np.around(out)
                        acc = np.mean(out[:, -random_length:] == target_output[:, -random_length:])
                        validation_loss.append(loss_value)
                        validation_acc.append(acc)
                    llprint("\tValidation Avg. Logistic Loss: %.4f, accuracy: %.4f\n" % (np.mean(validation_loss), np.mean(validation_acc)))
                    # print("out\n", out[:, -random_length:])
                    # print("target\n", target_output[:, -random_length:])
                    if np.mean(validation_loss) == 0.0:
                        exit()


                if take_checkpoint:
                    llprint("\nSaving Checkpoint ... "),
                    ncomputer.save(session, ckpts_dir, 'step-%d' % (i))
                    llprint("Done!\n")

# words_count = 15
# word_size = 10
# read_heads = 1

# allocation_gates
#  (batch_size, seq_length, 1)
# free_gates
#  (batch_size, seq_length, read_heads)
# read_weightings
#  (batch_size, seq_length, words_count, read_heads)
# usage_vectors
#  (batch_size, seq_length, words_count)
# write_gates
#  (batch_size, seq_length, 1)
# write_weightings
#  (batch_size, seq_length, words_count)

# allocation_gates
#  (1, 19, 1)
# free_gates
#  (1, 19, 1)
# read_weightings
#  (1, 19, 15, 1)
# usage_vectors
#  (1, 19, 15)
# write_gates
#  (1, 19, 1)
# write_weightings
#  (1, 19, 15)
