import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np
import pickle
import getopt
import time
import sys
import os

from dnc.dnc import DNC
from recurrent_controller import RecurrentController

def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()

def load(path):
    return pickle.load(open(path, 'rb'))

def onehot(index, size):
    vec = np.zeros(size, dtype=np.float32)
    vec[index] = 1.0
    return vec

def prepare_sample(sample, target_code, word_space_size):
    input_vec = np.array(sample[0]['inputs'], dtype=np.float32)
    output_vec = np.array(sample[0]['inputs'], dtype=np.float32)
    seq_len = input_vec.shape[0]
    weights_vec = np.zeros(seq_len, dtype=np.float32)

    target_mask = (input_vec == target_code)
    output_vec[target_mask] = sample[0]['outputs']
    weights_vec[target_mask] = 1.0

    input_vec = np.array([onehot(code, word_space_size) for code in input_vec])
    output_vec = np.array([onehot(code, word_space_size) for code in output_vec])

    return (
        np.reshape(input_vec, (1, -1, word_space_size)),
        np.reshape(output_vec, (1, -1, word_space_size)),
        seq_len,
        np.reshape(weights_vec, (1, -1, 1))
    )



if __name__ == '__main__':

    dirname = os.path.dirname(__file__)
    ckpts_dir = os.path.join(dirname , 'checkpoints')
    data_dir = os.path.join(dirname, 'data', 'en-10k')
    tb_logs_dir = os.path.join(dirname, 'logs')

    llprint("Loading Data ... ")
    lexicon_dict = load(os.path.join(data_dir, 'lexicon-dict.pkl'))
    data = load(os.path.join(data_dir, 'train', 'train.pkl'))
    llprint("Done!\n")

    batch_size = 1
    input_size = output_size = len(lexicon_dict)
    sequence_max_length = 100
    word_space_size = len(lexicon_dict)
    words_count = 256
    word_size = 64
    read_heads = 4

    learning_rate = 1e-4
    momentum = 0.9

    from_checkpoint = None
    iterations = 100000
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
                RecurrentController,
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
            loss = tf.reduce_mean(
                loss_weights * tf.nn.softmax_cross_entropy_with_logits(output, ncomputer.target_output)
            )

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

                    sample = np.random.choice(data, 1)
                    input_data, target_output, seq_len, weights = prepare_sample(sample, lexicon_dict['-'], word_space_size)

                    summerize = (i % 100 == 0)
                    take_checkpoint = (i != 0) and (i % end == 0)

                    loss_value, _, summary = session.run([
                        loss,
                        apply_gradients,
                        summerize_op if summerize else no_summerize
                    ], feed_dict={
                        ncomputer.input_data: input_data,
                        ncomputer.target_output: target_output,
                        ncomputer.sequence_length: seq_len,
                        loss_weights: weights
                    })

                    last_100_losses.append(loss_value)
                    summerizer.add_summary(summary, i)

                    if summerize:
                        llprint("\n\tAvg. Cross-Entropy: %.7f\n" % (np.mean(last_100_losses)))

                        end_time_100 = time.time()
                        elapsed_time = (end_time_100 - start_time_100) / 60
                        avg_counter += 1
                        avg_100_time += (1. / avg_counter) * (elapsed_time - avg_100_time)
                        estimated_time = (avg_100_time * ((end - i) / 100.)) / 60.

                        print "\tAvg. 100 iterations time: %.2f minutes" % (avg_100_time)
                        print "\tApprox. time to completion: %.2f hours" % (estimated_time)

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
