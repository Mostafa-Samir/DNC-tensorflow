import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np
import getopt
import sys
import os

from dnc.dnc import DNC
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

    dirname = os.path.dirname(__file__)
    ckpts_dir = os.path.join(dirname , 'checkpoints')
    tb_logs_dir = os.path.join(dirname, 'logs')

    batch_size = 1
    input_size = output_size = 6
    sequence_max_length = 10
    words_count = 15
    word_size = 10
    read_heads = 1

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

            optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=momentum)

            ncomputer = DNC(
                FeedforwardController,
                input_size,
                output_size,
                2 * sequence_max_length + 1,
                words_count,
                word_size,
                read_heads,
                batch_size
            )

            # squash the DNC output between 0 and 1
            output, _ = ncomputer.get_outputs()
            squashed_output = tf.clip_by_value(tf.sigmoid(output), 1e-6, 1. - 1e-6)

            loss = binary_cross_entropy(squashed_output, ncomputer.target_output)

            summeries = []

            gradients = optimizer.compute_gradients(loss)
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    summeries.append(tf.summary.histogram(var.name + '/grad', grad))
                    gradients[i] = (tf.clip_by_value(grad, -10, 10), var)

            apply_gradients = optimizer.apply_gradients(gradients)

            summeries.append(tf.summary.scalar("Loss", loss))

            summerize_op = tf.summary.merge(summeries)
            no_summerize = tf.no_op()

            summerizer = tf.summary.FileWriter(tb_logs_dir, session.graph)

            llprint("Done!\n")

            llprint("Initializing Variables ... ")
            session.run(tf.global_variables_initializer())
            llprint("Done!\n")

            if from_checkpoint is not None:
                llprint("Restoring Checkpoint %s ... " % (from_checkpoint))
                ncomputer.restore(session, ckpts_dir, from_checkpoint)
                llprint("Done!\n")


            last_100_losses = []

            for i in xrange(iterations + 1):
                llprint("\rIteration %d/%d" % (i, iterations))

                random_length = np.random.randint(1, sequence_max_length + 1)
                input_data, target_output = generate_data(batch_size, random_length, input_size)

                summerize = (i % 100 == 0)
                take_checkpoint = (i != 0) and (i % iterations == 0)

                loss_value, _, summary = session.run([
                    loss,
                    apply_gradients,
                    summerize_op if summerize else no_summerize
                ], feed_dict={
                    ncomputer.input_data: input_data,
                    ncomputer.target_output: target_output,
                    ncomputer.sequence_length: 2 * random_length + 1
                })

                last_100_losses.append(loss_value)
                summerizer.add_summary(summary, i)

                if summerize:
                    llprint("\n\tAvg. Logistic Loss: %.4f\n" % (np.mean(last_100_losses)))
                    last_100_losses = []

                if take_checkpoint:
                    llprint("\nSaving Checkpoint ... "),
                    ncomputer.save(session, ckpts_dir, 'step-%d' % (i))
                    llprint("Done!\n")
