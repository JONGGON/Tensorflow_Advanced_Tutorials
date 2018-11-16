import os
import shutil

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm


def model(TEST=False, optimizer_selection="Adam", learning_rate=0.001, training_epochs=10,
          batch_size=256, display_step=1, batch_norm=False, regularization='L1', scale=0.0001):
    mnist = input_data.read_data_sets("", one_hot=True)

    model_name = "CNN"
    if batch_norm == True:
        model_name = "BN" + model_name
    else:
        if regularization == "L1" or regularization == "L2":
            model_name = "reg" + regularization + model_name

    if TEST == False:
        if os.path.exists("tensorboard/{}".format(model_name)):
            shutil.rmtree("tensorboard/{}".format(model_name))

    # stride? -> [1, 2, 2, 1] = [one image, width, height, one channel]
    def final_conv2d(input, weight_shape='', bias_shape='', strides=[1, 1, 1, 1], padding="VALID"):

        # weight_init = tf.contrib.layers.xavier_initializer(uniform=False)
        weight_init = tf.truncated_normal_initializer(stddev=0.02)
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
        conv_out = tf.nn.conv2d(input, w, strides=strides, padding=padding)

        return tf.nn.bias_add(conv_out, b)

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

    # ksize, strides? -> [1, 2, 2, 1] = [one image, width, height, one channel]
    # pooling을 할때, 각 batch 에 대해 한 채널에 대해서 하니까, 1, 1,로 설정해준것.
    def pooling(input, type="avg", k=2, padding='VALID'):
        if type == "max":
            return tf.nn.max_pool(input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding=padding)
        else:
            return tf.nn.avg_pool(input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding=padding)

    def inference(x):

        # ReceptiveField_inspection폴더의 rf.py에서 확인해보면, ReceptiveField사이즈가 28임을 알 수 있 습니다.

        x = tf.reshape(x, shape=[-1, 28, 28, 1])
        # in pooling, type = max or avg
        with tf.variable_scope("conv_1"):
            conv_1 = tf.nn.leaky_relu(
                conv2d(x, weight_shape=[5, 5, 1, 24], bias_shape=[24], strides=[1, 1, 1, 1], padding="VALID"))
            pool_1 = pooling(conv_1, type="avg", k=2, padding="VALID")  # result -> batch_size, 12, 12, 12
        with tf.variable_scope("conv_2"):
            conv_2 = tf.nn.leaky_relu(
                conv2d(pool_1, weight_shape=[5, 5, 24, 48], bias_shape=[48], strides=[1, 1, 1, 1], padding="VALID"))
            pool_2 = pooling(conv_2, type="avg", k=2, padding="VALID")  # result -> batcj_size, 4, 4 ,24
        with tf.variable_scope("conv_3"):
            conv_3 = tf.nn.leaky_relu(
                conv2d(pool_2, weight_shape=[4, 4, 48, 72], bias_shape=[72], strides=[1, 1, 1, 1], padding="VALID"))
            # result -> batcj_size, 1, 1 ,36
        with tf.variable_scope("conv_4"):
            output = final_conv2d(conv_3, weight_shape=[1, 1, 72, 10], bias_shape=[10], strides=[1, 1, 1, 1],
                                  padding="VALID")
        return tf.reshape(output, (-1, 10))

    def loss(output, y):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y)
        loss = tf.reduce_mean(cross_entropy)
        return loss

    def training(cost, global_step):
        tf.summary.scalar("cost", cost)
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

    def evaluate(output, y):
        correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(output), 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy

    # print(tf.get_default_graph()) #기본그래프이다.
    JG_Graph = tf.Graph()  # 내 그래프로 설정한다.- 혹시라도 나중에 여러 그래프를 사용할 경우를 대비
    with JG_Graph.as_default():  # as_default()는 JG_Graph를 기본그래프로 설정한다.
        with tf.name_scope("feed_dict"):
            x = tf.placeholder("float", [None, 784])
            y = tf.placeholder("float", [None, 10])
        with tf.variable_scope("shared_variables", reuse=tf.AUTO_REUSE) as scope:
            with tf.name_scope("inference"):
                output = inference(x)
            # or scope.reuse_variables()

        # optimizer의 매개변수들을 저장하고 싶지 않다면 여기에 선언해야한다.
        with tf.name_scope("saver"):
            saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=3)
        if not TEST:
            with tf.name_scope("loss"):
                global_step = tf.Variable(0, name="global_step", trainable=False)
                cost = loss(output, y)
            with tf.name_scope("trainer"):
                train_operation = training(cost, global_step)
            with tf.name_scope("tensorboard"):
                summary_operation = tf.summary.merge_all()

        with tf.name_scope("evaluation"):
            evaluate_operation = evaluate(output, y)

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

            for epoch in tqdm(range(1, training_epochs + 1)):
                print("epoch : {}".format(epoch))
                avg_cost = 0.
                total_batch = int(mnist.train.num_examples / batch_size)

                for i in range(total_batch):
                    mbatch_x, mbatch_y = mnist.train.next_batch(batch_size)
                    feed_dict = {x: mbatch_x, y: mbatch_y}
                    sess.run(train_operation, feed_dict=feed_dict)

                    minibatch_cost = sess.run(cost, feed_dict=feed_dict)
                    avg_cost += (minibatch_cost / total_batch)

                print("cost : {0:0.3}".format(avg_cost))
                if epoch % display_step == 0:
                    val_feed_dict = {x: mnist.validation.images, y: mnist.validation.labels}
                    accuracy = sess.run(evaluate_operation, feed_dict=val_feed_dict)
                    print("Validation Accuracy : {0:0.3f}".format(100 * accuracy))

                    summary_str = sess.run(summary_operation, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, global_step=sess.run(global_step))

                    save_model_path = os.path.join('model', model_name)
                    if not os.path.exists(save_model_path):
                        os.makedirs(save_model_path)
                    saver.save(sess, save_model_path + '/', global_step=sess.run(global_step),
                               write_meta_graph=False)

            print("Optimization Finished!")

        # batch_norm=True 일 때, 이동평균 사용
        if TEST:
            test_feed_dict = {x: mnist.test.images, y: mnist.test.labels}
            accuracy = sess.run(evaluate_operation, feed_dict=test_feed_dict)
            print("Test Accuracy : {0:0.3f}".format(100 * accuracy))


if __name__ == "__main__":
    # optimizers_ selection = "Adam" or "RMSP" or "SGD"
    # batch normalization은 Hidden Layer에만 추가합니다. 또한 활성화 함수전에 적용합니다.
    # regularization -> batch_norm = False 일때, L2 or L1 or nothing
    model(TEST=True, optimizer_selection="Adam", learning_rate=0.001, training_epochs=10,
          batch_size=256, display_step=1, batch_norm=True, regularization='L2', scale=0.0001)
else:
    print("model imported")
