import os
import shutil
from collections import *

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm

import PCA


def model(TEST=True, Comparison_with_PCA=True, model_name="Autoencoder", target_sparsity=0.1, weight_sparsity=0.2,
          optimizer_selection="Adam",
          learning_rate=0.001, training_epochs=100,
          batch_size=128, display_step=10, batch_norm=True, regularization='L1', scale=0.0001):
    mnist = input_data.read_data_sets("", one_hot=False)

    if batch_norm == True:
        model_name = "BN" + model_name
    else:
        if regularization == "L1" or regularization == "L2":
            model_name =  regularization + model_name

    if TEST == False:
        if os.path.exists("tensorboard/{}".format(model_name)):
            shutil.rmtree("tensorboard/{}".format(model_name))

    def final_layer(input, weight_shape, bias_shape):

        weight_init = tf.random_normal_initializer(stddev=0.01)
        bias_init = tf.random_normal_initializer(stddev=0.01)
        weight_decay = tf.constant(scale, dtype=tf.float32)
        if regularization == "L1":
            w = tf.get_variable("w", weight_shape, initializer=weight_init,
                                regularizer=tf.contrib.layers.l1_regularizer(scale=weight_decay))
        elif regularization == "L2":
            w = tf.get_variable("w", weight_shape, initializer=weight_init,
                                regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay))
        else:
            w = tf.get_variable("w", weight_shape, initializer=weight_init)
        b = tf.get_variable("b", bias_shape, initializer=bias_init)

        return tf.matmul(input, w) + b

    def layer(input, weight_shape, bias_shape):

        weight_init = tf.truncated_normal_initializer(stddev=0.02)
        bias_init = tf.truncated_normal_initializer(stddev=0.02)
        if batch_norm:
            w = tf.get_variable("w", weight_shape, initializer=weight_init)
        else:
            weight_decay = tf.constant(scale, dtype=tf.float32)
            if regularization == "L1":
                w = tf.get_variable("w", weight_shape, initializer=weight_init,
                                    regularizer=tf.contrib.layers.l1_regularizer(scale=weight_decay))
            elif regularization == "L2":
                w = tf.get_variable("w", weight_shape, initializer=weight_init,
                                    regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay))
            else:
                w = tf.get_variable("w", weight_shape, initializer=weight_init)
        b = tf.get_variable("b", bias_shape, initializer=bias_init)

        if batch_norm:
            return tf.layers.batch_normalization(tf.matmul(input, w) + b, training=not TEST)
        else:
            return tf.matmul(input, w) + b

    # stride? -> [1, 2, 2, 1] = [one image, width, height, one channel]
    def conv2d(input, weight_shape='', bias_shape='', strides=[1, 1, 1, 1], padding="VALID"):
        # weight_init = tf.contrib.layers.xavier_initializer(uniform=False)
        weight_init = tf.truncated_normal_initializer(stddev=0.02)
        bias_init = tf.constant_initializer(value=0)
        if batch_norm:
            w = tf.get_variable("w", weight_shape, initializer=weight_init)
        else:
            weight_decay = tf.constant(scale, dtype=tf.float32)
            if regularization == "L1":
                w = tf.get_variable("w", weight_shape, initializer=weight_init,
                                    regularizer=tf.contrib.layers.l1_regularizer(scale=weight_decay))
            elif regularization == "L2":
                w = tf.get_variable("w", weight_shape, initializer=weight_init,
                                    regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay))
            else:
                w = tf.get_variable("w", weight_shape, initializer=weight_init)

        b = tf.get_variable("b", bias_shape, initializer=bias_init)
        conv_out = tf.nn.conv2d(input, w, strides=strides, padding=padding)

        if batch_norm:
            return tf.layers.batch_normalization(tf.nn.bias_add(conv_out, b), training=not TEST)
        else:
            return tf.nn.bias_add(conv_out, b)

    def final_transpose_conv2d(input, output_shape='', weight_shape='', bias_shape='', strides=[1, 1, 1, 1],
                               padding="VALID"):
        weight_init = tf.contrib.layers.xavier_initializer(uniform=False)
        bias_init = tf.constant_initializer(value=0)
        weight_decay = tf.constant(scale, dtype=tf.float32)
        if regularization == "L1":
            w = tf.get_variable("w", weight_shape, initializer=weight_init,
                                regularizer=tf.contrib.layers.l1_regularizer(scale=weight_decay))
        elif regularization == "L2":
            w = tf.get_variable("w", weight_shape, initializer=weight_init,
                                regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay))
        else:
            w = tf.get_variable("w", weight_shape, initializer=weight_init)

        b = tf.get_variable("b", bias_shape, initializer=bias_init)

        conv_out = tf.nn.conv2d_transpose(input, w, output_shape=output_shape, strides=strides, padding=padding)
        return tf.nn.bias_add(conv_out, b)

    def conv2d_transpose(input, output_shape='', weight_shape='', bias_shape='', strides=[1, 1, 1, 1], padding="VALID"):
        weight_init = tf.contrib.layers.xavier_initializer(uniform=False)
        bias_init = tf.constant_initializer(value=0)
        if batch_norm:
            w = tf.get_variable("w", weight_shape, initializer=weight_init)
        else:
            weight_decay = tf.constant(scale, dtype=tf.float32)
            if regularization == "L1":
                w = tf.get_variable("w", weight_shape, initializer=weight_init,
                                    regularizer=tf.contrib.layers.l1_regularizer(scale=weight_decay))
            elif regularization == "L2":
                w = tf.get_variable("w", weight_shape, initializer=weight_init,
                                    regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay))
            else:
                w = tf.get_variable("w", weight_shape, initializer=weight_init)

        b = tf.get_variable("b", bias_shape, initializer=bias_init)

        conv_out = tf.nn.conv2d_transpose(input, w, output_shape=output_shape, strides=strides, padding=padding)
        if batch_norm:
            return tf.layers.batch_normalization(tf.nn.bias_add(conv_out, b), training=not TEST)
        else:
            return tf.nn.bias_add(conv_out, b)

    def inference(x):
        hidden = []
        if model_name == "Autoencoder" or model_name == "BNAutoencoder" or model_name == "L1Autoencoder" or model_name == "L2Autoencoder":
            with tf.variable_scope("encoder"):
                with tf.variable_scope("fully1"):
                    fully_1 = tf.nn.sigmoid(layer(tf.reshape(x, (-1, 784)), [784, 256], [256]))
                    hidden.append(fully_1)
                with tf.variable_scope("fully2"):
                    fully_2 = tf.nn.sigmoid(layer(fully_1, [256, 128], [128]))
                    hidden.append(fully_2)
                with tf.variable_scope("fully3"):
                    fully_3 = tf.nn.sigmoid(layer(fully_2, [128, 64], [64]))
                    hidden.append(fully_3)
                with tf.variable_scope("output"):
                    encoder_output = tf.nn.sigmoid(layer(fully_3, [64, 2], [2]))

            with tf.variable_scope("decoder"):
                with tf.variable_scope("fully1"):
                    fully_4 = tf.nn.sigmoid(layer(encoder_output, [2, 64], [64]))
                    hidden.append(fully_4)
                with tf.variable_scope("fully2"):
                    fully_5 = tf.nn.sigmoid(layer(fully_4, [64, 128], [128]))
                    hidden.append(fully_5)
                with tf.variable_scope("fully3"):
                    fully_6 = tf.nn.sigmoid(layer(fully_5, [128, 256], [256]))
                    hidden.append(fully_6)
                with tf.variable_scope("output"):
                    decoder_output = tf.nn.sigmoid(final_layer(fully_6, [256, 784], [784]))
            return encoder_output, decoder_output, hidden

        elif model_name == 'Convolution_Autoencoder' or model_name == "BNConvolution_Autoencoder" or model_name == 'L1Convolution_Autoencoder' or model_name == "L2Convolution_Autoencoder":
            with tf.variable_scope("encoder"):
                with tf.variable_scope("conv_1"):
                    conv_1 = tf.nn.sigmoid(
                        conv2d(x, weight_shape=[5, 5, 1, 32], bias_shape=[32], strides=[1, 1, 1, 1], padding="VALID"))
                    # result -> batch_size, 24, 24, 32
                    hidden.append(conv_1)
                with tf.variable_scope("conv_2"):
                    conv_2 = tf.nn.sigmoid(
                        conv2d(conv_1, weight_shape=[5, 5, 32, 32], bias_shape=[32], strides=[1, 1, 1, 1],
                               padding="VALID"))
                    hidden.append(conv_2)
                    # result -> batch_size, 20, 20, 32
                with tf.variable_scope("conv_3"):
                    conv_3 = tf.nn.sigmoid(
                        conv2d(conv_2, weight_shape=[5, 5, 32, 32], bias_shape=[32], strides=[1, 1, 1, 1],
                               padding="VALID"))
                    hidden.append(conv_3)
                    # result -> batch_size, 16, 16, 32
                with tf.variable_scope("conv_4"):
                    conv_4 = tf.nn.sigmoid(
                        conv2d(conv_3, weight_shape=[5, 5, 32, 32], bias_shape=[32], strides=[1, 1, 1, 1],
                               padding="VALID"))
                    hidden.append(conv_4)
                    # result -> batch_size, 12, 12, 32
                with tf.variable_scope("conv_5"):
                    conv_5 = tf.nn.sigmoid(
                        conv2d(conv_4, weight_shape=[5, 5, 32, 32], bias_shape=[32], strides=[1, 1, 1, 1],
                               padding="VALID"))
                    # result -> batch_size, 8, 8, 32
                with tf.variable_scope("conv_6"):
                    conv_6 = tf.nn.sigmoid(
                        conv2d(conv_5, weight_shape=[5, 5, 32, 32], bias_shape=[32], strides=[1, 1, 1, 1],
                               padding="VALID"))
                    hidden.append(conv_5)
                    # result -> batch_size, 4, 4, 32
                with tf.variable_scope("output"):
                    encoder_output = tf.nn.sigmoid(
                        conv2d(conv_6, weight_shape=[4, 4, 32, 2], bias_shape=[2], strides=[1, 1, 1, 1],
                               padding="VALID"))
                    hidden.append(conv_6)
                    # result -> batch_size, 1, 1, 2

            with tf.variable_scope("decoder"):
                with tf.variable_scope("trans_conv_1"):
                    conv_7 = tf.nn.sigmoid(
                        conv2d_transpose(encoder_output, output_shape=tf.shape(conv_6), weight_shape=[4, 4, 32, 2],
                                         bias_shape=[32], strides=[1, 1, 1, 1], padding="VALID"))
                    hidden.append(conv_7)
                    # result -> batch_size, 4, 4, 32
                with tf.variable_scope("trans_conv_2"):
                    conv_8 = tf.nn.sigmoid(
                        conv2d_transpose(conv_7, output_shape=tf.shape(conv_5), weight_shape=[5, 5, 32, 32],
                                         bias_shape=[32],
                                         strides=[1, 1, 1, 1], padding="VALID"))
                    hidden.append(conv_8)
                    # result -> batch_size, 8, 8, 32
                with tf.variable_scope("trans_conv_3"):
                    conv_9 = tf.nn.sigmoid(
                        conv2d_transpose(conv_8, output_shape=tf.shape(conv_4), weight_shape=[5, 5, 32, 32],
                                         bias_shape=[32],
                                         strides=[1, 1, 1, 1], padding="VALID"))
                    hidden.append(conv_9)
                    # result -> batch_size, 12, 12, 32
                with tf.variable_scope("trans_conv_4"):
                    conv_10 = tf.nn.sigmoid(
                        conv2d_transpose(conv_9, output_shape=tf.shape(conv_3), weight_shape=[5, 5, 32, 32],
                                         bias_shape=[32],
                                         strides=[1, 1, 1, 1], padding="VALID"))
                    hidden.append(conv_10)
                    # result -> batch_size, 16, 16, 32
                with tf.variable_scope("trans_conv_5"):
                    conv_11 = tf.nn.sigmoid(
                        conv2d_transpose(conv_10, output_shape=tf.shape(conv_2), weight_shape=[5, 5, 32, 32],
                                         bias_shape=[32],
                                         strides=[1, 1, 1, 1], padding="VALID"))
                    hidden.append(conv_11)
                    # result -> batch_size, 20, 20, 32
                with tf.variable_scope("trans_conv_6"):
                    conv_12 = tf.nn.sigmoid(
                        conv2d_transpose(conv_11, output_shape=tf.shape(conv_1), weight_shape=[5, 5, 32, 32],
                                         bias_shape=[32],
                                         strides=[1, 1, 1, 1], padding="VALID"))
                    hidden.append(conv_12)
                    # result -> batch_size, 24, 24, 32

                with tf.variable_scope("output"):
                    decoder_output = tf.nn.sigmoid(
                        final_transpose_conv2d(conv_12, output_shape=tf.shape(x), weight_shape=[5, 5, 1, 32],
                                               bias_shape=[1],
                                               strides=[1, 1, 1, 1], padding="VALID"))
                    # result -> batch_size, 28, 28, 1
            return encoder_output, decoder_output, hidden

    def evaluate(output, x):
        with tf.variable_scope("validation"):
            tf.summary.image('input_image', tf.reshape(x, [-1, 28, 28, 1]), max_outputs=5)
            tf.summary.image('output_image', tf.reshape(output, [-1, 28, 28, 1]), max_outputs=5)

            if model_name == 'Convolution_Autoencoder' or model_name == "BNConvolution_Autoencoder" or model_name == 'L1Convolution_Autoencoder' or model_name == "L2Convolution_Autoencoder":
                l2 = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(output, x)), axis=[1, 2, 3]))
            elif model_name == "Autoencoder" or model_name == "BNAutoencoder" or model_name == "L1Autoencoder" or model_name == "L2Autoencoder":
                l2 = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(output, tf.reshape(x, (-1, 784)))), axis=1))

            val_loss = tf.reduce_mean(l2)
            tf.summary.scalar('val_cost', val_loss)
            return val_loss

    def training(cost, global_step):
        tf.summary.scalar("train_cost", cost)
        if not batch_norm:
            cost = tf.add_n([cost] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            if optimizer_selection == "Adam":
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            elif optimizer_selection == "RMSP":
                optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            elif optimizer_selection == "SGD":
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            train_operation = optimizer.minimize(cost, global_step=global_step)
        return train_operation

    def loss(output, x):
        if model_name == 'Convolution_Autoencoder' or model_name == "BNConvolution_Autoencoder" or model_name == 'L1Convolution_Autoencoder' or model_name == "L2Convolution_Autoencoder":
            l2 = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(output, x)), axis=[1, 2, 3]))
        elif model_name == "Autoencoder" or model_name == "BNAutoencoder" or model_name == "L1Autoencoder" or model_name == "L2Autoencoder":
            l2 = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(output, tf.reshape(x, (-1, 784)))), axis=1))
        train_loss = tf.reduce_mean(l2)
        return train_loss

    # 각 뉴런이 활성화되었을 때와 그렇지 않을 때 두 가지 경우만 있으므로, 아래와 같이 나눠 쓸 수 있다고한다.
    def KLD(tp, rp):  # target probability , real probability
        return tp * tf.log(tp / (rp + 1e-8)) + (1 - tp) * tf.log((1 - tp) / ((1 - rp) + 1e-8))

    # print(tf.get_default_graph()) #기본그래프이다.
    JG_Graph = tf.Graph()  # 내 그래프로 설정한다.- 혹시라도 나중에 여러 그래프를 사용할 경우를 대비
    with JG_Graph.as_default():  # as_default()는 JG_Graph를 기본그래프로 설정한다.
        with tf.name_scope("feed_dict"):
            x = tf.placeholder("float", [None, 28, 28, 1])
        with tf.variable_scope("shared_variables", reuse=tf.AUTO_REUSE) as scope:
            with tf.name_scope("inference"):
                encoder_output, decoder_output, hidden = inference(x)
            # or scope.reuse_variables()

        # optimizer의 매개변수들을 저장하고 싶지 않다면 여기에 선언해야한다.
        with tf.name_scope("saver"):
            saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=3)
        if not TEST:
            with tf.name_scope("loss"):
                global_step = tf.Variable(0, name="global_step", trainable=False)
                cost = loss(decoder_output, x)
                '''
                1. 각 뉴런에 대해 평균 sparsity loss를 구한다 -> 평균 값을 구하기 위해 batchsize로 나눈다.
                2. 위에서 나온 sparsity loss 를 다 더한다.
                3. cost에 더한다.
                '''
                # Sparse Autoencoder loss
                for h in hidden:
                    temp_h = tf.reduce_mean(weight_sparsity * KLD(target_sparsity, h), axis=0)
                    cost += tf.reduce_sum(temp_h)

            with tf.name_scope("trainer"):
                train_operation = training(cost, global_step)
            with tf.name_scope("tensorboard"):
                summary_operation = tf.summary.merge_all()

        with tf.name_scope("evaluation"):
            evaluate_operation = evaluate(decoder_output, x)

    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(graph=JG_Graph, config=config) as sess:
        print("initializing!!!")
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(os.path.join('model', model_name))
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Restore {} checkpoint!!!".format(os.path.basename(ckpt.model_checkpoint_path)))
            saver.restore(sess, ckpt.model_checkpoint_path)
            # shutil.rmtree("model/{}/".format(model_name))

        if not TEST:

            summary_writer = tf.summary.FileWriter(os.path.join("tensorboard", model_name), sess.graph)

            for epoch in tqdm(range(training_epochs)):
                avg_cost = 0.
                total_batch = int(mnist.train.num_examples / batch_size)
                for i in range(total_batch):
                    mbatch_x, mbatch_y = mnist.train.next_batch(batch_size)
                    feed_dict = {x: mbatch_x.reshape((-1, 28, 28, 1))}
                    _, minibatch_cost = sess.run([train_operation, cost], feed_dict=feed_dict)
                    avg_cost += (minibatch_cost / total_batch)

                print("cost : {}".format(avg_cost))
                if epoch % display_step == 0:
                    val_feed_dict = {x: mnist.validation.images[:1000].reshape(
                        (-1, 28, 28, 1))}  # GPU 메모리 인해 mnist.test.images[:1000], 여기서 1000이다.
                    val_cost, summary_str = sess.run([evaluate_operation, summary_operation],
                                                     feed_dict=val_feed_dict)
                    print("Validation L2 cost : {}".format(val_cost))
                    summary_writer.add_summary(summary_str, global_step=sess.run(global_step))

                    save_model_path = os.path.join('model', model_name)
                    if not os.path.exists(save_model_path):
                        os.makedirs(save_model_path)
                    saver.save(sess, save_model_path + '/', global_step=sess.run(global_step),
                               write_meta_graph=False)

            print("Optimization Finished!")

        # batch_norm=True 일 때, 이동평균 사용
        if Comparison_with_PCA and TEST:
            # PCA , Autoencoder Visualization
            test_feed_dict = {x: mnist.test.images.reshape(-1, 28, 28,
                                                           1)}  # GPU 메모리 인해 mnist.test.images[:1000], 여기서 1000이다.
            pca_applied = PCA.PCA(n_components=2, show_reconstruction_image=False)  # 10000,2
            encoder_applied, test_cost = sess.run([encoder_output, evaluate_operation],
                                                  feed_dict=test_feed_dict)
            print("Test L2 cost : {}".format(test_cost))
            applied = OrderedDict(PCA=pca_applied, Autoencoder=encoder_applied.reshape(-1, 2))

            # PCA , Autoencoder 그리기
            fig, ax = plt.subplots(1, 2, figsize=(18, 12))
            # fig.suptitle('vs', size=20, color='r')
            for x, (key, value) in enumerate(applied.items()):
                ax[x].grid(False)
                ax[x].set_title(key, size=20, color='k')
                ax[x].set_axis_off()
                for num in range(10):
                    ax[x].scatter(
                        [value[:, 0][i] for i in range(len(mnist.test.labels)) if mnist.test.labels[i] == num], \
                        [value[:, 1][j] for j in range(len(mnist.test.labels)) if mnist.test.labels[j] == num], \
                        s=10, label=str(num), marker='o')
                ax[x].legend()

            # plt.tight_layout()
            if model_name == "Autoencoder":
                plt.savefig("PCA vs Autoencoder.png", dpi=300)
            elif model_name == "BNAutoencoder":
                plt.savefig("PCA vs batchAutoencoder.png", dpi=300)
            elif model_name == "L1Autoencoder":
                plt.savefig("PCA vs L1Autoencoder.png", dpi=300)
            elif model_name == "L2Autoencoder":
                plt.savefig("PCA vs L2Autoencoder.png", dpi=300)

            elif model_name == "Convolution_Autoencoder":
                plt.savefig("PCA vs ConvAutoencoder.png", dpi=300)
            elif model_name == "BNConvolution_Autoencoder":
                plt.savefig("PCA vs BNConvAutoencoder.png", dpi=300)
            elif model_name == "L1Autoencoder":
                plt.savefig("PCA vs L1Autoencoder.png", dpi=300)
            elif model_name == "L2Convolution_Autoencoder":
                plt.savefig("PCA vs L2ConvAutoencoder.png", dpi=300)
            plt.show()


if __name__ == "__main__":
    # optimizers_ selection = "Adam" or "RMSP" or "SGD"
    # model_name = "Convolution_Autoencoder" or "Autoencoder"
    # batch normalization은 Hidden Layer에만 추가합니다. 또한 활성화 함수전에 적용합니다.
    # regularization -> batch_norm = False 일때, L2 or L1 or nothing
    model(TEST=True, Comparison_with_PCA=True, model_name="Autoencoder", target_sparsity=0.2, weight_sparsity=0.1,
          optimizer_selection="Adam", learning_rate=0.001, training_epochs=1, batch_size=512, display_step=1,
          batch_norm=False, regularization='L1', scale=0.0001)
else:
    print("model imported")
