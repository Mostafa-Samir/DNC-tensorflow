import tensorflow as tf

from data_sequences import TwoNumbersConstDataProvider
from tqdm import tqdm
import numpy as np

from .settings import set_dict


n_classes = 10
forget_bias = 1.0
max_inputs_length = set_dict["max_inputs_length"]
batch_size = set_dict["batch_size"]
inputs = tf.placeholder(tf.int8, [None, max_inputs_length * 2 + 1, n_classes])
targets = tf.placeholder(tf.int8, [None, max_inputs_length + 1, n_classes])

n_hidden = 256
with tf.variable_scope("fw_pass"):
    inputs = tf.cast(inputs, tf.float32)
    cell = tf.nn.rnn_cell.LSTMCell(n_hidden)
    _, states = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)

with tf.variable_scope("bw_pass") as bw_scope:
    go_symbol = tf.zeros(
        [batch_size, 1, n_classes], dtype=tf.float32, name="GO")
    w_conv = tf.get_variable("w_conv", shape=[n_hidden, n_classes])
    bw_cell = tf.nn.rnn_cell.LSTMCell(n_hidden)
    inner_state = states
    output = tf.squeeze(go_symbol)
    outputs_list = []
    for i in range(max_inputs_length + 1):
        if i > 0:
            bw_scope.reuse_variables()
        output, inner_state = bw_cell(output, inner_state, scope=bw_scope)
        output = tf.matmul(output, w_conv)
        outputs_list.append(output)
    output = tf.pack(outputs_list)
    output = tf.transpose(output, [1, 0, 2])

targets = tf.cast(targets, tf.float32)
targets_for_loss = tf.reshape(targets, [-1, n_classes])
output = tf.reshape(output, [-1, n_classes])
cross_enthropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    output, targets_for_loss))
weight_decay = 1e-4
reg_loss = l2_loss = weight_decay * tf.add_n(
    [tf.nn.l2_loss(var) for var in tf.trainable_variables()])
# optimizer = tf.train.AdamOptimizer(learning_rate=set_dict["lr"])
optimizer = tf.train.MomentumOptimizer(
    set_dict["lr"], set_dict["momentum"], use_nesterov=True)
train_op = optimizer.minimize(cross_enthropy + reg_loss)
prediction = tf.nn.softmax(output)
correct_prediction = tf.equal(
    tf.argmax(prediction, 1),
    tf.argmax(targets_for_loss, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar("cross_enthropy", cross_enthropy)
tf.summary.scalar("reg_loss", reg_loss)
tf.summary.scalar("accuracy", accuracy)

if __name__ == '__main__':
    VALIDATE = True
    VALIDETE_PRINT = False
    data_provider = TwoNumbersConstDataProvider(
        max_inputs_length=max_inputs_length,
        train_size=set_dict["train_size"],
        valid_size=set_dict["valid_size"],
        test_size=set_dict["test_size"],
        delimiter=-1,
        use_cache=True,
        one_hot=True,
        shuffle=True)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session:
        merged = tf.summary.merge_all()
        summ_dir = 'logs'
        ending = '_lstm_max_inp_length_{}_train_size_{}'.format(max_inputs_length, set_dict["train_size"])
        train_writer = tf.summary.FileWriter(summ_dir + '/train_%s' % ending)
        test_writer = tf.summary.FileWriter(summ_dir + '/test_%s' % ending)
        session.run(tf.global_variables_initializer())
        for epoch in range(set_dict["epochs"]):
            print("\n", '-' * 30, "Epoch: %d" % epoch, '-' * 30, '\n')
            print("Training")
            train_cross_enthropy = []
            train_mean_accuracy = []
            train_mean_reg_loss = []
            # for i in tqdm(range(data_provider.train.num_examples // batch_size)):
            for i in range(data_provider.train.num_examples // batch_size):
                _, _, inputs_concat_batch, targets_batch = data_provider.train.next_batch(
                    batch_size)
                inputs_concat_batch = np.flip(inputs_concat_batch, axis=1)
                targets_batch = np.flip(targets_batch, axis=1)
                fetches = [train_op, cross_enthropy, accuracy, merged, reg_loss]
                feed_dict = {
                    inputs: inputs_concat_batch,
                    targets: targets_batch
                }
                _, loss_fetched, acc_fetched, summary, reg_loss_fetched = session.run(
                    fetches, feed_dict=feed_dict)
                train_cross_enthropy.append(loss_fetched)
                train_mean_accuracy.append(acc_fetched)
                train_mean_reg_loss.append(reg_loss_fetched)
                train_writer.add_summary(summary, epoch * batch_size + 1)
            print("Mean loss: ", np.mean(train_cross_enthropy),
                  " mean reg loss", np.mean(train_mean_reg_loss),
                  " mean accurracy: ", np.mean(train_mean_accuracy))

            if VALIDATE:
                valid_loss = []
                valid_acc = []
                print("validation")
                # for i in tqdm(range(data_provider.valid.num_examples // batch_size)):
                for i in range(data_provider.valid.num_examples // batch_size):
                    inp_1, inp_2, inputs_concat_batch, targets_batch = data_provider.valid.next_batch(
                        batch_size)
                    inputs_concat_batch = np.flip(inputs_concat_batch, axis=1)
                    targets_batch = np.flip(targets_batch, axis=1)
                    fetches = [cross_enthropy, accuracy, prediction, merged]
                    feed_dict = {
                        inputs: inputs_concat_batch,
                        targets: targets_batch
                    }
                    loss_fetched, acc_fetched, pred, summary = session.run(
                        fetches, feed_dict=feed_dict)
                    valid_loss.append(loss_fetched)
                    valid_acc.append(acc_fetched)
                    test_writer.add_summary(summary, epoch * batch_size + 1)
                print("Mean loss: ", np.mean(valid_loss), " mean accurracy: ", np.mean(valid_acc))
                if VALIDETE_PRINT:
                    targets_batch = np.flip(targets_batch, axis=1)
                    pred = np.reshape(pred, [batch_size, max_inputs_length + 1, n_classes])
                    pred = np.flip(pred, axis=1)
                    for i in range(10):
                        print(
                            "target:    ",
                            np.argmax(targets_batch[i], axis=1),
                            "||",
                            ''.join(str(a) for a in np.argmax(inp_1[i], axis=1)),
                            "+",
                            ''.join(str(a) for a in np.argmax(inp_2[i], axis=1)))
                        print("prediction:", np.argmax(pred[i], axis=1))
