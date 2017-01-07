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


def onehot(x, n):
    ret = np.zeros(n).astype(np.float32)
    ret[x] = 1.0
    return ret


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
    sums = np.array(sums)
    sums = sums.reshape(1, 1, 1)

    return x_seq_list, sums, sums_text


def cross_entropy(prediction, target):
    return tf.square(prediction - target)


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
            summerizer = tf.summary.FileWriter(tb_logs_dir, session.graph)

            ncomputer = DNC(
                FeedforwardController,
                input_size,
                output_size,
                sequence_max_length + 1,
                words_count,
                word_size,
                read_heads,
                batch_size
            )

            # squash the DNC output to a scalar number
            squashed_target_output = tf.reduce_sum(ncomputer.target_output)
            output, _ = ncomputer.get_outputs()
            squashed_output = tf.reduce_sum(output)

            loss = cross_entropy(squashed_output, squashed_target_output)
            gradients = optimizer.compute_gradients(loss)
            apply_gradients = optimizer.apply_gradients(gradients)

            summerize_loss = tf.summary.scalar("Loss", loss)

            summerize_op = tf.summary.merge([summerize_loss])
            no_summerize = tf.no_op()

            llprint("Done!\n")

            llprint("Initializing Variables ... ")
            session.run(tf.global_variables_initializer())
            llprint("Done!\n")

            if from_checkpoint is not None:
                llprint("Restoring Checkpoint %s ... " % (from_checkpoint))
                ncomputer.restore(session, ckpts_dir, from_checkpoint)
                llprint("Done!\n")

            last_100_losses = []

            for i in range(iterations + 1):
                # We use for training just (sequence_max_length / 10) examples
                random_length = np.random.randint(2, (sequence_max_length / 10) + 1)
                input_data, target_output, sums_text = generate_data(random_length, input_size)

                summerize = (i % 100 == 0)
                take_checkpoint = (i != 0) and (i % iterations == 0)

                target_out, output, loss_value, _, summary = session.run([
                    squashed_target_output,
                    squashed_output,
                    loss,
                    apply_gradients,
                    summerize_op if summerize else no_summerize
                ], feed_dict={
                    ncomputer.input_data: input_data,
                    ncomputer.target_output: target_output,
                    ncomputer.sequence_length: random_length + 1
                })

                last_100_losses.append(loss_value)
                summerizer.add_summary(summary, i)

                if summerize:
                    llprint("\rIteration %d/%d" % (i, iterations))
                    llprint("\nAvg. Logistic Loss: %.4f\n" % (np.mean(last_100_losses)))
                    print "Real value: ", sums_text[:-2] + ' = ' + str(int(target_output[0]))
                    print "Predicted:  ", sums_text[:-2] + ' = ' + str(int(output // 1)) + " [" + str(output) + "]"
                    last_100_losses = []

                if take_checkpoint:
                    llprint("\nSaving Checkpoint ... "),
                    ncomputer.save(session, ckpts_dir, 'step-%d' % (i))
                    llprint("Done!\n")

            llprint("\nTesting generalization...\n")
            for i in xrange(iterations + 1):
                llprint("\nIteration %d/%d" % (i, iterations))
                # We test now the learned generalization using sequence_max_length examples
                random_length = np.random.randint(2, sequence_max_length + 1)
                input_data, target_output, sums_text = generate_data(random_length, input_size)

                target_out, output = session.run([
                    squashed_target_output,
                    squashed_output
                ], feed_dict={
                    ncomputer.input_data: input_data,
                    ncomputer.target_output: target_output,
                    ncomputer.sequence_length: random_length + 1
                })

                print "\nReal value: ", sums_text[:-2] + ' = ' + str(int(target_output[0]))
                print "Predicted:  ", sums_text[:-2] + ' = ' + str(int(output // 1)) + " [" + str(output) + "]"