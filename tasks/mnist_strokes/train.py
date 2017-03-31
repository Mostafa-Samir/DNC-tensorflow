from __future__ import print_function

import time
import sys
import os
import sys

import tensorflow as tf
import numpy as np
from tqdm import tqdm
from data_sequences import MNISTStrokesDataProvider

from dnc.dnc import DNC
from mnist_controller import LSTMController, FeedforwardController


def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()


def _get_next_batch_from_data(data, batch_size, num_digits, padding):
    total_inputs = []
    total_targets = []
    for batch_part_idx in range(batch_size):
        batch_part = data.next_batch(num_digits)
        _, inputs_slice, _, labels_slice = batch_part
        # part_inputs = batch_part[1]
        new_part_inputs = []
        for inp in inputs_slice:
            inp[0] = np.zeros(4)
            new_part_inputs.extend(inp)
        # add <EOS> symbol filled with ones
        new_part_inputs.append(np.ones(4))
        new_part_inputs = np.array(new_part_inputs)
        total_inputs.append(new_part_inputs)

        # part_targets = batch_part[3]
        total_targets.append(labels_slice)
    total_targets = np.array(total_targets)

    # pad inputs for the same shape
    inputs_length = [inp.shape[0] for inp in total_inputs]
    max_length = max(inputs_length)
    numpy_inputs = np.zeros((batch_size, max_length, 4))
    for inp_idx, inp in enumerate(total_inputs):
        if padding == 'right':
            numpy_inputs[inp_idx, :inputs_length[inp_idx]] = inp
        elif padding == 'left':
            numpy_inputs[inp_idx, -inputs_length[inp_idx]:] = inp
        else:
            raise Exception("Unsoported type of padding")

    return numpy_inputs, total_targets, inputs_length


def binary_cross_entropy(predictions, targets):

    return tf.reduce_mean(
        -1 * targets * tf.log(predictions) - (1 - targets) * tf.log(1 - predictions)
    )


epochs_dict = {
    5: 25,
    10: 35,
    15: 80
}


if __name__ == '__main__':
    batch_size = 20
    num_digits = 15
    epochs = epochs_dict[num_digits]
    # real max - 117, mean abt 40
    print("{} digits, {} epochs".format(num_digits, epochs))
    entries_per_digit = 100
    max_seq_length = entries_per_digit * num_digits
    data_provider = MNISTStrokesDataProvider(
        validation_split=0.1,
        one_hot=True)
    print("Validation examples: ", data_provider.validation.num_examples)
    # train 54000
    # test 10000
    # _input, target, seq_length = _get_next_batch_from_data(data_provider.train, batch_size, num_digits)
    # import ipdb; ipdb.set_trace()
    # print("_input", _input.shape)
    # print("target", target.shape)
    # print("seq_length", np.array(seq_length).shape)
    # ('_input', (1, 182, 4))
    # ('target', (1, 5, 10))
    # ('seq_length', (1,))


    dirname = os.path.dirname(__file__)
    ckpts_dir = os.path.join(dirname , 'checkpoints')
    tb_logs_dir = os.path.join(dirname, 'logs')

    input_size = 4
    output_size = 10
    # settings for memory from previous run
    # words_count = 25
    # word_size = 20
    # read_heads = 4
    words_count = 256 // 4
    word_size = 64 // 4
    read_heads = 4
    print("words count", words_count, "word size", word_size, "read heads", read_heads)


    learning_rate = 1e-4
    # learning_rate = 1e-2
    momentum = 0.9

    iterations = 100000

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session:

        llprint("Building Computational Graph ... ")

        optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=momentum)

        ncomputer = DNC(
            # FeedforwardController,
            LSTMController,
            input_size,
            output_size,
            max_seq_length,
            # # Use default params
            words_count,  # 256
            word_size,  # 64
            read_heads, # 4
            batch_size=batch_size,
            req_out_seq_length=num_digits,
        )

        # squash the DNC output between 0 and 1
        output, _ = ncomputer.get_outputs()
        # get only last N entries - predicted numbers
        output = tf.slice(output, [0, ncomputer.sequence_length, 0], [-1, -1, -1])
        # output = tf.squeeze(output, axis=0)
        # targ = tf.squeeze(ncomputer.target_output, axis=0)
        targ = ncomputer.target_output
        print("output", output.get_shape())
        prediction = tf.nn.softmax(output)
        # prediction = tf.
        out_argmax = tf.argmax(prediction, 2)
        print("out_argmax", out_argmax.get_shape())
        targ_argmax = tf.argmax(targ, 2)
        print("targ_argmax", targ_argmax.get_shape())
        correct_prediction = tf.equal(out_argmax, targ_argmax)
        accuracy = tf.reduce_mean(tf.cast(tf.squeeze(tf.reshape(correct_prediction, [-1])), tf.float32))
        # squashed_output = tf.clip_by_value(tf.sigmoid(output), 1e-6, 1. - 1e-6)

        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(output, targ)
        )
        # loss = binary_cross_entropy(tf.squeeze(prediction, axis=0), tf.squeeze(ncomputer.target_output, axis=0))

        summeries = []

        gradients = optimizer.compute_gradients(loss)
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                # summeries.append(tf.histogram_summary(var.name + '/grad', grad))
                gradients[i] = (tf.clip_by_value(grad, -10, 10), var)
        apply_gradients = optimizer.apply_gradients(gradients)

        # apply_gradients = optimizer.minimize(loss)

        summeries.append(tf.scalar_summary("Loss", loss))
        summeries.append(tf.scalar_summary("Accuracy", accuracy))

        summerize_op = tf.merge_summary(summeries)
        no_summerize = tf.no_op()

        summerizer = tf.train.SummaryWriter(tb_logs_dir, session.graph)

        llprint("Done!\n")

        llprint("Initializing Variables ... ")
        session.run(tf.initialize_all_variables())
        llprint("Done!\n")

        for epoch in range(epochs):
            print("\n", '-' * 30, "Epoch: %d" % epoch, '-' * 30, '\n')

            train_loss = []
            train_acc = []
            print("training")
            for i in tqdm(range(data_provider.train.num_examples // num_digits // batch_size)):
            # for i in tqdm(range(10)):
                # llprint("\rIteration %d/%d" % (i, iterations))

                summerize = (i % 10 == 0)
                # take_checkpoint = (i != 0) and (i % iterations == 0)
                _input, target, seq_length = _get_next_batch_from_data(
                    data_provider.train, batch_size, num_digits, padding='left')

                loss_value, _, summary, pred, acc = session.run([
                    loss,
                    apply_gradients,
                    summerize_op if summerize else no_summerize,
                    prediction,
                    accuracy,
                ], feed_dict={
                    ncomputer.input_data: _input,
                    ncomputer.target_output: target,
                    # ncomputer.sequence_length: seq_length[0]
                    ncomputer.sequence_length: max(seq_length)
                })

                train_loss.append(loss_value)
                train_acc.append(acc)
                summerizer.add_summary(summary, i)

                # if summerize:
                #     llprint("\n\tAvg. Logistic Loss: %.4f, accuracy: %.5f\n" % (np.mean(last_100_losses), acc))
                #     print("prediction full\n", pred)
                #     print("prediction", np.argmax(pred, axis=1))
                #     print("target", np.argmax(target, axis=2)[0])
                #     last_100_losses = []

                # if take_checkpoint:
                #     llprint("\nSaving Checkpoint ... "),
                #     ncomputer.save(session, ckpts_dir, 'step-%d' % (i))
                #     llprint("Done!\n")
            print("Mean loss: ", np.mean(train_loss), " mean accurracy: ", np.mean(train_acc))

            # testing
            valid_loss = []
            valid_acc = []
            print("validation")
            for i in tqdm(range(data_provider.validation.num_examples // num_digits // batch_size)):
            # for i in tqdm(range(10)):
                _input, target, seq_length = _get_next_batch_from_data(
                    data_provider.validation, batch_size, num_digits,
                    padding='left')

                loss_value, acc = session.run([
                    loss,
                    accuracy,
                ], feed_dict={
                    ncomputer.input_data: _input,
                    ncomputer.target_output: target,
                    # ncomputer.sequence_length: seq_length[0]
                    ncomputer.sequence_length: max(seq_length)
                })

                valid_loss.append(loss_value)
                valid_acc.append(acc)
            print("Mean loss: ", np.mean(valid_loss), " mean accurracy: ", np.mean(valid_acc))


