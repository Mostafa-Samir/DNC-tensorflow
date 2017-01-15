import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np
import getopt
import sys
import time
import os

from dnc.dnc import DNC
from feedforward_controller import FeedforwardController


def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()


def onehot(index, size):
    vec = np.zeros(size, dtype=np.float32)
    vec[index] = 1.0
    return vec


def generate_data(length, size):

    content = np.random.randint(0, size - 1, length)

    seqlen = length + 1
    x_seq_list = [float('nan')] * seqlen
    sums = 0.0
    sums_text = ""
    for i in range(seqlen):
        if (i < length):
            x_seq_list[i] = onehot(content[i], size)
            sums += content[i]
            sums_text += str(content[i]) + " + "
        else:
            x_seq_list[i] = onehot(size - 1, size)

    x_seq_list = np.array(x_seq_list)
    x_seq_list = x_seq_list.reshape((1,) + x_seq_list.shape)
    x_seq_list = np.reshape(x_seq_list, (1, -1, size))

    target_output = np.zeros((1, 1, seqlen), dtype=np.float32)
    target_output[:, -1, -1] = sums
    target_output = np.reshape(target_output, (1, -1, 1))

    weights_vec = np.zeros((1, 1, seqlen), dtype=np.float32)
    weights_vec[:, -1, -1] = 1.0
    weights_vec = np.reshape(weights_vec, (1, -1, 1))

    return x_seq_list, target_output, sums_text, weights_vec


if __name__ == '__main__':

    dirname = os.path.dirname(__file__)
    ckpts_dir = os.path.join(dirname , 'checkpoints')
    tb_logs_dir = os.path.join(dirname, 'logs')

    batch_size = 1
    input_size = 3
    output_size = 1
    sequence_max_length = 100
    words_count = 15
    word_size = 10
    read_heads = 1

    learning_rate = 1e-4
    momentum = 0.9

    from_checkpoint = None
    iterations = 1000
    start_step = 0

    options,_ = getopt.getopt(sys.argv[1:], '', ['checkpoint=', 'iterations=', 'start='])

    for opt in options:
        if opt[0] == '--checkpoint':
            from_checkpoint = opt[1]
        elif opt[0] == '--iterations':
            iterations = int(opt[1])
        elif opt[0] == '--start':
            start_step = int(opt[1])

    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(graph=graph) as session:

            llprint("Building Computational Graph ... ")

            optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=momentum)
            summerizer = tf.train.SummaryWriter(tb_logs_dir, session.graph)

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

            output, _ = ncomputer.get_outputs()

            loss_weights = tf.placeholder(tf.float32, [batch_size, None, 1])
            loss = tf.reduce_mean(tf.square((loss_weights * output) - ncomputer.target_output))

            summeries = []

            gradients = optimizer.compute_gradients(loss)
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    gradients[i] = (tf.clip_by_value(grad, -10, 10), var)
            for (grad, var) in gradients:
                if grad is not None:
                    summeries.append(tf.histogram_summary(var.name + '/grad', grad))

            apply_gradients = optimizer.apply_gradients(gradients)

            summeries.append(tf.scalar_summary("Loss", loss))

            summerize_op = tf.merge_summary(summeries)
            no_summerize = tf.no_op()

            llprint("Done!\n")

            llprint("Initializing Variables ... ")
            session.run(tf.initialize_all_variables())
            llprint("Done!\n")

            if from_checkpoint is not None:
                llprint("Restoring Checkpoint %s ... " % (from_checkpoint))
                ncomputer.restore(session, ckpts_dir, from_checkpoint)
                llprint("Done!\n")


            last_100_losses = []

            start = 0 if start_step == 0 else start_step + 1
            end = start_step + iterations + 1

            start_time_100 = time.time()
            end_time_100 = None
            avg_100_time = 0.
            avg_counter = 0

            for i in xrange(start, end + 1):
                try:
                    llprint("\rIteration %d/%d" % (i, end))

                    # We use for training just (sequence_max_length / 2) examples
                    seq_len = np.random.randint(2, (sequence_max_length / 10) + 1)
                    input_data, target_output, sums_text, weights = generate_data(seq_len, input_size)

                    summerize = (i % 100 == 0)
                    take_checkpoint = (i != 0) and (i % end == 0)

                    output_, loss_value, _, summary = session.run([
                        output,
                        loss,
                        apply_gradients,
                        summerize_op if summerize else no_summerize
                    ], feed_dict={
                        ncomputer.input_data: input_data,
                        ncomputer.target_output: target_output,
                        ncomputer.sequence_length: seq_len + 1,
                        loss_weights: weights
                    })

                    last_100_losses.append(loss_value)
                    summerizer.add_summary(summary, i)

                    if summerize:
                        llprint("\n\nAvg. Cross-Entropy: %.7f\n" % (np.mean(last_100_losses)))

                        end_time_100 = time.time()
                        elapsed_time = (end_time_100 - start_time_100) / 60
                        avg_counter += 1
                        avg_100_time += (1. / avg_counter) * (elapsed_time - avg_100_time)
                        estimated_time = (avg_100_time * ((end - i) / 100.)) / 60.

                        print "Avg. 100 iterations time: %.2f minutes" % (avg_100_time)
                        print "Approx. time to completion: %.2f hours" % (estimated_time)
                        print "DNC input", input_data
                        print "Text input:     ", sums_text[:-2] + " = ? "
                        print "Target_output", target_output
                        output_ = output_ * weights
                        print "DNC output", output_
                        print "Real operation:   ", sums_text[:-2] + ' = ' + str(int(target_output[-1, -1, -1]))
                        print "Predicted result: ", sums_text[:-2] + ' = ' + str(int(round(output_[-1, -1, -1]))) + " [" + str(output_[-1, -1, -1]) + "]"

                        start_time_100 = time.time()
                        last_100_losses = []

                    if take_checkpoint:
                        llprint("\nSaving Checkpoint ... "),
                        ncomputer.save(session, ckpts_dir, 'step-%d' % (i))
                        llprint("Done!\n")
                except KeyboardInterrupt:

                    llprint("\nSaving Checkpoint ... "),
                    ncomputer.save(session, ckpts_dir, 'step-%d' % (i))
                    llprint("Done!\n")
                    sys.exit(0)

            llprint("\nTesting generalization...\n")
            for i in xrange(50):
                llprint("\nIteration %d/%d" % (i, iterations))
                # We test now the learned generalization using sequence_max_length examples
                seq_len = np.random.randint(2, sequence_max_length + 1)
                input_data, target_output, sums_text, weights = generate_data(seq_len, input_size)

                output_ = session.run([output], feed_dict={
                        ncomputer.input_data: input_data,
                        ncomputer.target_output: target_output,
                        ncomputer.sequence_length: seq_len + 1,
                        loss_weights: weights
                })

                output_ = output_ * weights

                print "\nReal operation:   ", sums_text[:-2] + ' = ' + str(int(target_output[-1, -1, -1]))
                print "Predicted result: ", sums_text[:-2] + ' = ' + str(int(round(output_[-1, -1, -1]))) + " [" + str(output_[-1, -1, -1]) + "]"

            llprint("Done!\n")
