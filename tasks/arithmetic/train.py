from __future__ import print_function

import time
import sys
import os
import sys
import argparse

import tensorflow as tf
import numpy as np
from tqdm import tqdm
from data_sequences import TwoNumbersConstDataProvider

from dnc.dnc import DNC
from controller import LSTMController, FeedforwardController
from settings import set_dict


def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()


def binary_cross_entropy(predictions, targets):

    return tf.reduce_mean(
        -1 * targets * tf.log(predictions) - (1 - targets) * tf.log(1 - predictions)
    )

def svm_loss(predictions, targets):
    # prediction = tf.reshape(predictions, [-1, 10])
    # target = tf.reshape(targets, [-1, 10])
    loss = tf.reduce_mean(tf.maximum(0.0, 1 + predictions - targets[tf.argmax(targets, axis=2)]))
    return loss

def one_hot_encoding(data):
    initial_shape = list(data.shape)
    data = data.reshape(-1)
    data = TwoNumbersConstDataProvider.labels_to_one_hot(data, 10)
    initial_shape.append(10)
    data = data.reshape(initial_shape)
    return data


def get_next_batch(dataset, batch_size):
    inputs_1, inputs_2, inputs_concat, targets = dataset.next_batch(batch_size)
    inputs_1 = one_hot_encoding(inputs_1)
    inputs_2 = one_hot_encoding(inputs_2)
    targets = one_hot_encoding(targets)
    st = np.zeros((batch_size, 1, 10))
    inputs = np.concatenate((inputs_1, st, inputs_2), axis=1)
    return inputs, targets


if __name__ == '__main__':
    VALIDATE = False
    VALIDETE_PRINT = True

    max_inputs_length = set_dict["max_inputs_length"]
    train_size = set_dict["train_size"]
    valid_size = set_dict["valid_size"]
    test_size = set_dict["test_size"]
    batch_size = set_dict["batch_size"]
    epochs = set_dict["epochs"]
    n_classes = 10
    input_size = 10
    output_size = 10
    words_count = 32
    word_size = 16
    read_heads = 1

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", '--controller', choices=["LSTM", "FC"], required=True)
    args = parser.parse_args()
    controller_name = args.controller
    controllers = {
        "FC": FeedforwardController,
        "LSTM": LSTMController,
    }


    learning_rate = set_dict["lr"]
    momentum = 0.9

    data_provider = TwoNumbersConstDataProvider(
        max_inputs_length=max_inputs_length,
        train_size=set_dict["train_size"],
        valid_size=set_dict["valid_size"],
        test_size=set_dict["test_size"],
        delimiter=-1,
        use_cache=True,
        one_hot=True,
        shuffle=True,)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session:

        llprint("Building Computational Graph with %s controller ... " % controller_name)

        # optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=momentum)

        ncomputer = DNC(
            controllers[controller_name],
            input_size,
            output_size,
            # max_inputs_length * 2 + 1,
            max_inputs_length * 3 + 2,
            words_count,  # 256
            word_size,  # 64
            read_heads,  # 4
            batch_size=batch_size,
            # req_out_seq_length=max_inputs_length + 1,
        )

        # # squash the DNC output between 0 and 1
        # output, _ = ncomputer.get_outputs()
        # print("output not sliced", output.get_shape())
        # # get only last N entries - predicted numbers
        # output = tf.slice(output, [0, ncomputer.sequence_length, 0], [-1, -1, -1])
        # print("output sliced", output.get_shape())
        # # # output = tf.squeeze(output, axis=0)
        # # # targ = tf.squeeze(ncomputer.target_output, axis=0)
        # # targ = ncomputer.target_output
        # # print("output", output.get_shape())
        # # prediction = tf.nn.softmax(output)
        # # out_argmax = tf.argmax(prediction, 2)
        # # print("out_argmax", out_argmax.get_shape())
        # # targ_argmax = tf.argmax(targ, 2)
        # # print("targ_argmax", targ_argmax.get_shape())
        # # correct_prediction = tf.equal(out_argmax, targ_argmax)
        # # accuracy = tf.reduce_mean(tf.cast(tf.squeeze(tf.reshape(correct_prediction, [-1])), tf.float32))
        # # # squashed_output = tf.clip_by_value(tf.sigmoid(output), 1e-6, 1. - 1e-6)

        # # loss = tf.reduce_mean(
        # #     tf.nn.softmax_cross_entropy_with_logits(output, targ)
        # # )
        # # # loss = binary_cross_entropy(tf.squeeze(prediction, axis=0), tf.squeeze(ncomputer.target_output, axis=0))
        # # gradients = optimizer.compute_gradients(loss)
        # # for i, (grad, var) in enumerate(gradients):
        # #     if grad is not None:
        # #         # summeries.append(tf.histogram_summary(var.name + '/grad', grad))
        # #         gradients[i] = (tf.clip_by_value(grad, -10, 10), var)
        # # apply_gradients = optimizer.apply_gradients(gradients)

        # # # apply_gradients = optimizer.minimize(loss)

        # targets = tf.cast(ncomputer.target_output, tf.float32)
        # print("targets", targets.get_shape())
        # targets_for_loss = tf.reshape(targets, [-1, n_classes])
        # output = tf.reshape(output, [-1, n_classes])
        # print("output reshaped", output.get_shape())
        # # loss = tf.nn.softmax_cross_entropy_with_logits(
        # #     output, targets_for_loss)
        # # optimizer = tf.train.AdamOptimizer(learning_rate=set_dict["lr"])
        # # apply_gradients = optimizer.minimize(loss)
        # # mean_loss = tf.reduce_mean(loss)
        # # prediction = tf.nn.softmax(output)
        # # correct_prediction = tf.equal(
        # #     tf.argmax(prediction, 1),
        # #     tf.argmax(targets_for_loss, 1))
        # # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


        # # tf.summary.scalar("loss", mean_loss)
        # # tf.summary.scalar("accuracy", accuracy)
        # cross_enthropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        #     output, targets_for_loss))

        output, _ = ncomputer.get_outputs()
        squashed_output = tf.clip_by_value(tf.sigmoid(output), 1e-6, 1. - 1e-6)
        squashed_output = tf.slice(squashed_output, [0, max_inputs_length * 2 + 1, 0], [-1, -1, -1])
        target_output = tf.slice(ncomputer.target_output, [0, max_inputs_length * 2 + 1, 0], [-1, -1, -1])
        cross_enthropy = binary_cross_entropy(squashed_output, target_output)
        # cross_enthropy = binary_cross_entropy(squashed_output, ncomputer.target_output)
        # cross_enthropy = tf.reduce_mean(tf.contrib.losses.hinge_loss(squashed_output, ncomputer.target_output))
        print("output", output.get_shape())

        weight_decay = 1e-4
        reg_loss = l2_loss = weight_decay * tf.add_n(
            [tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        # optimizer = tf.train.AdamOptimizer(learning_rate=set_dict["lr"])
        optimizer = tf.train.MomentumOptimizer(
            set_dict["lr"], set_dict["momentum"], use_nesterov=True)
        # train_op = optimizer.minimize(cross_enthropy + reg_loss)
        gradients = optimizer.compute_gradients(cross_enthropy + reg_loss)
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                # summeries.append(tf.histogram_summary(var.name + '/grad', grad))
                gradients[i] = (tf.clip_by_value(grad, -10, 10), var)
        train_op = optimizer.apply_gradients(gradients)

        prediction = tf.nn.softmax(output)
        print("prediction", prediction.get_shape())
        prediction = tf.reshape(prediction, [-1, n_classes])
        print("ncomputer.target_output", ncomputer.target_output.get_shape())
        targets_for_loss = tf.reshape(ncomputer.target_output, [-1, n_classes])
        correct_prediction = tf.equal(
            tf.argmax(prediction, 1),
            tf.argmax(targets_for_loss, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar("cross_enthropy", cross_enthropy)
        tf.summary.scalar("reg_loss", reg_loss)
        tf.summary.scalar("accuracy", accuracy)

        summerize_op = tf.summary.merge_all()

        llprint("Done!\n")

        llprint("Initializing Variables ... ")
        summ_dir = 'logs'
        ending = '_dnc_updated_{}_controller_max_inp_length_{}_train_size_{}'.format(controller_name, max_inputs_length, set_dict["train_size"])
        train_writer = tf.summary.FileWriter(summ_dir + '/train_%s' % ending)
        test_writer = tf.summary.FileWriter(summ_dir + '/test_%s' % ending)
        session.run(tf.global_variables_initializer())
        session.run(tf.initialize_all_variables())
        llprint("Done!\n")

        for epoch in range(epochs):
            print("\n", '-' * 30, "Epoch: %d" % epoch, '-' * 30, '\n')
            start_time = time.time()
            train_loss = []
            train_acc = []
            train_reg_loss = []
            print("training")
            # for i in tqdm(range(data_provider.train.num_examples // batch_size)):
            for i in range(data_provider.train.num_examples // batch_size):

                _, _, inputs_concat_batch, targets_batch = data_provider.train.next_batch(
                    batch_size)
                # print("inputs_concat_batch", inputs_concat_batch.shape)
                # print("targets_batch", targets_batch.shape)
                inputs_concat_batch = np.flip(inputs_concat_batch, axis=1)
                inputs_concat_batch_initial_shape = inputs_concat_batch.shape
                st = np.zeros((batch_size, targets_batch.shape[1], 10))
                st = st + 1
                inputs_concat_batch = np.concatenate((inputs_concat_batch, st), axis=1)
                st_targets = np.zeros((batch_size, inputs_concat_batch_initial_shape[1], 10))
                targets_batch = np.concatenate((st_targets, targets_batch), axis=1)

                # targets_batch = np.flip(targets_batch, axis=1)
                loss_value, reg_fetched, _, summary, pred, acc = session.run([
                    cross_enthropy,
                    reg_loss,
                    train_op,
                    summerize_op,
                    prediction,
                    accuracy,
                ], feed_dict={
                    ncomputer.input_data: inputs_concat_batch,
                    ncomputer.target_output: targets_batch,
                    # ncomputer.sequence_length: max_inputs_length * 2 + 1,
                    ncomputer.sequence_length: max_inputs_length * 3 + 2,
                })

                train_loss.append(loss_value)
                train_acc.append(acc)
                train_reg_loss.append(reg_fetched)
                train_writer.add_summary(summary, epoch * batch_size + 1)

            time_cons = time.time() - start_time
            print("Mean loss: ", np.mean(train_loss),
                  " mean reg loss", np.mean(train_reg_loss),
                  " mean accurracy: ", np.mean(train_acc),
                  " time pre epoch: ", time_cons)
            pred = np.reshape(pred, [batch_size, -1, n_classes])
            for i in range(min(5, batch_size)):
                print(
                    "target:    ",
                    np.argmax(targets_batch[i], axis=1))
                print("prediction:", np.argmax(pred[i], axis=1))


            if VALIDATE:
                # testing
                valid_loss = []
                valid_acc = []
                print("validation")
                # for i in tqdm(range(data_provider.valid.num_examples // batch_size)):
                for i in range(data_provider.valid.num_examples // batch_size):
                    inp_1, inp_2, inputs_concat_batch, targets_batch = data_provider.valid.next_batch(
                        batch_size)
                    # targets_batch = np.flip(targets_batch, axis=1)
                    inputs_concat_batch = np.flip(inputs_concat_batch, axis=1)
                    inputs_concat_batch_initial_shape = inputs_concat_batch.shape
                    st = np.zeros((batch_size, targets_batch.shape[1], 10))
                    inputs_concat_batch = np.concatenate((inputs_concat_batch, st), axis=1)
                    st_targets = np.zeros((batch_size, inputs_concat_batch_initial_shape[1], 10))
                    targets_batch = np.concatenate((st_targets, targets_batch), axis=1)

                    loss_value, acc, summary, pred = session.run([
                        cross_enthropy,
                        accuracy,
                        summerize_op,
                        prediction,
                    ], feed_dict={
                        ncomputer.input_data: inputs_concat_batch,
                        ncomputer.target_output: targets_batch,
                        # ncomputer.sequence_length: max_inputs_length * 2 + 1,
                        ncomputer.sequence_length: max_inputs_length * 3 + 2,
                    })

                    valid_loss.append(loss_value)
                    valid_acc.append(acc)
                    test_writer.add_summary(summary, epoch * batch_size + 1)
                print("Mean loss: ", np.mean(valid_loss), " mean accurracy: ", np.mean(valid_acc))
                if VALIDETE_PRINT:
                    # targets_batch = np.flip(targets_batch, axis=1)
                    pred = np.reshape(pred, [batch_size, -1, n_classes])
                    # pred = np.flip(pred, axis=1)
                    for i in range(min(5, batch_size)):
                        print(
                            "target:    ",
                            np.argmax(targets_batch[i], axis=1),
                            "||",
                            ''.join(str(a) for a in np.argmax(inp_1[i], axis=1)),
                            "+",
                            ''.join(str(a) for a in np.argmax(inp_2[i], axis=1)))
                        print("prediction:", np.argmax(pred[i], axis=1))
