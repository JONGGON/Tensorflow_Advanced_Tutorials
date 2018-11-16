import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm


# evaluate the data
def show_image(model_name, generated_image, column_size=10, row_size=10):
    print("show image")
    '''generator image visualization'''
    fig_g, ax_g = plt.subplots(row_size, column_size, figsize=(column_size, row_size))
    fig_g.suptitle('MNIST_generator')
    for j in range(row_size):
        for i in range(column_size):
            ax_g[j][i].grid(False)
            ax_g[j][i].set_axis_off()
            ax_g[j][i].imshow(generated_image[i + j * column_size].reshape((28, 28)), cmap='gray')
    fig_g.savefig("{}_generator.png".format(model_name))
    plt.show()


def model(TEST=True, noise_size=100, targeting=True, distance_loss="L2", distance_loss_weight=1,
          optimizer_selection="Adam", learning_rate=0.001, training_epochs=100,
          batch_size=128, display_step=10, batch_norm=True, regularization='L1', scale=0.0001):
    mnist = input_data.read_data_sets("", one_hot=True)

    if targeting == False:
        print("random generative GAN")
        model_name = "GeneralGAN"

    else:
        if distance_loss == "L1":
            print("target generative GAN with L1 loss")
            model_name = "CGANL1"
        elif distance_loss == "L2":
            print("target generative GAN with L2 loss")
            model_name = "CGANL2"
        else:
            print("target generative GAN")
            model_name = "CGAN"

    if batch_norm == True:
        model_name = "BN" + model_name
    else:
        if regularization == "L1" or regularization == "L2":
            model_name = "reg" + regularization + model_name

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

    def generator(noise=None, target=None):
        if targeting:
            noise = tf.concat([noise, target], axis=1)
        with tf.variable_scope("generator"):
            with tf.variable_scope("fully1"):
                fully_1 = tf.nn.leaky_relu(layer(noise, [np.shape(noise)[1], 256], [256]))
            with tf.variable_scope("fully2"):
                fully_2 = tf.nn.leaky_relu(layer(fully_1, [256, 512], [512]))
            with tf.variable_scope("output"):
                output = tf.nn.sigmoid(final_layer(fully_2, [512, 784], [784]))

        return output

    def discriminator(x=None, target=None):
        if targeting:
            x = tf.concat([x, target], axis=1)
        with tf.variable_scope("discriminator"):
            with tf.variable_scope("fully1"):
                fully_1 = tf.nn.leaky_relu(layer(x, [np.shape(x)[1], 500], [500]))
            with tf.variable_scope("fully2"):
                fully_2 = tf.nn.leaky_relu(layer(fully_1, [500, 100], [100]))
            with tf.variable_scope("output"):
                output = final_layer(fully_2, [100, 1], [1])
        return output, tf.nn.sigmoid(output)

    def training(cost, var_list, scope=None):

        '''
        GAN 구현시 Batch Normalization을 쓸 때 주의할 점!!!
        #scope를 써줘야 한다. - 그냥 tf.get_collection(tf.GraphKeys.UPDATE_OPS) 이렇게 써버리면 
        shared_variables 아래에 있는 변수들을 다 업데이트 하므로 scope를 지정해줘야한다.
        '''
        if not batch_norm:
            cost = tf.add_n([cost] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=scope))
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)):
            if optimizer_selection == "Adam":
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            elif optimizer_selection == "RMSP":
                optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            elif optimizer_selection == "SGD":
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            train_operation = optimizer.minimize(cost, var_list=var_list)
        return train_operation

    def min_max_loss(logits=None, labels=None):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

    # print(tf.get_default_graph()) #기본그래프이다.
    JG_Graph = tf.Graph()  # 내 그래프로 설정한다.- 혹시라도 나중에 여러 그래프를 사용할 경우를 대비
    with JG_Graph.as_default():  # as_default()는 JG_Graph를 기본그래프로 설정한다.
        with tf.name_scope("feed_dict"):
            x = tf.placeholder("float", [None, 784])
            target = tf.placeholder("float", [None, 10])
            z = tf.placeholder("float", [None, noise_size])
        with tf.variable_scope("shared_variables", reuse=tf.AUTO_REUSE) as scope:
            with tf.name_scope("generator"):
                G = generator(noise=z, target=target)
            with tf.name_scope("discriminator"):
                D_real, sigmoid_D_real = discriminator(x=x, target=target)
                # scope.reuse_variables()
                D_gene, sigmoid_D_gene = discriminator(x=G, target=target)

        # Algorithjm
        var_D = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='shared_variables/discriminator')

        var_G = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='shared_variables/generator')

        # optimizer의 매개변수들을 저장하고 싶지 않다면 여기에 선언해야한다.
        with tf.name_scope("saver"):
            saver_all = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=3)
            saver_generator = tf.train.Saver(var_list=var_G, max_to_keep=3)

        if not TEST:

            # Algorithjm - 속이고 속이는 과정

            with tf.name_scope("Discriminator_loss"):
                # for discriminator
                D_Loss = min_max_loss(logits=D_real, labels=tf.ones_like(D_real)) + min_max_loss(logits=D_gene,
                                                                                                 labels=tf.zeros_like(
                                                                                                     D_gene))
                tf.summary.scalar("Discriminator Loss", D_Loss)

            with tf.name_scope("Generator_loss"):
                # for generator
                G_Loss = min_max_loss(logits=D_gene, labels=tf.ones_like(D_gene))
                tf.summary.scalar("Generator Loss", G_Loss)

            if distance_loss == "L1":
                with tf.name_scope("L1_loss"):
                    dis_loss = tf.losses.absolute_difference(x, G)
                    tf.summary.scalar("{} Loss".format(distance_loss), dis_loss)
                    G_Loss += tf.multiply(dis_loss, distance_loss_weight)
            elif distance_loss == "L2":
                with tf.name_scope("L1_loss"):
                    dis_loss = tf.losses.mean_squared_error(x, G)
                    tf.summary.scalar("{} Loss".format(distance_loss), dis_loss)
                    G_Loss += tf.multiply(dis_loss, distance_loss_weight)
            else:
                dis_loss = tf.constant(value=0, dtype=tf.float32)

            with tf.name_scope("Discriminator_trainer"):
                D_train_op = training(D_Loss, var_D, scope='shared_variables/discriminator')
            with tf.name_scope("Generator_trainer"):
                G_train_op = training(G_Loss, var_G, scope='shared_variables/generator')

            summary_operation = tf.summary.merge_all()

    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(graph=JG_Graph, config=config) as sess:
        print("initializing!!!")
        sess.run(tf.global_variables_initializer())
        ckpt_all = tf.train.get_checkpoint_state(os.path.join(model_name, 'All'))
        ckpt_generator = tf.train.get_checkpoint_state(os.path.join(model_name, 'Generator'))
        if (ckpt_all and tf.train.checkpoint_exists(ckpt_all.model_checkpoint_path)) \
                or (ckpt_generator and tf.train.checkpoint_exists(ckpt_generator.model_checkpoint_path)):
            if not TEST:
                print("all variable retored except for optimizer parameter")
                print("Restore {} checkpoint!!!".format(os.path.basename(ckpt_all.model_checkpoint_path)))
                saver_all.restore(sess, ckpt_all.model_checkpoint_path)
            else:
                print("generator variable retored except for optimizer parameter")
                print("Restore {} checkpoint!!!".format(os.path.basename(ckpt_generator.model_checkpoint_path)))
                saver_generator.restore(sess, ckpt_generator.model_checkpoint_path)

        if not TEST:
            summary_writer = tf.summary.FileWriter(os.path.join("tensorboard", model_name), sess.graph)

            for epoch in tqdm(range(1, training_epochs + 1)):

                Loss_D = 0.
                Loss_G = 0
                Loss_Distance = 0
                # 아래의 두 값이 각각 0.5 씩을 갖는게 가장 이상적이다.
                sigmoid_D = 0
                sigmoid_G = 0

                total_batch = int(mnist.train.num_examples / batch_size)
                for i in range(total_batch):
                    mbatch_x, mbatch_y = mnist.train.next_batch(batch_size)
                    noise = np.random.normal(loc=0.0, scale=1.0, size=(batch_size, noise_size))
                    feed_dict_all = {x: mbatch_x, target: mbatch_y, z: noise}
                    feed_dict_Generator = {x: mbatch_x, target: mbatch_y, z: noise}
                    _, Discriminator_Loss, D_real_simgoid = sess.run([D_train_op, D_Loss, sigmoid_D_real],
                                                                     feed_dict=feed_dict_all)
                    _, Generator_Loss, Distance_Loss, D_gene_simgoid = sess.run(
                        [G_train_op, G_Loss, dis_loss, sigmoid_D_gene],
                        feed_dict=feed_dict_Generator)
                    Loss_D += (Discriminator_Loss / total_batch)
                    Loss_G += (Generator_Loss / total_batch)
                    Loss_Distance += (Distance_Loss / total_batch)
                    sigmoid_D += D_real_simgoid / total_batch
                    sigmoid_G += D_gene_simgoid / total_batch

                print("Discriminator mean output : {} / Generator mean output : {}".format(np.mean(sigmoid_D),
                                                                                           np.mean(sigmoid_G)))

                if distance_loss == "L1" or distance_loss == "L2":
                    print(
                        "Discriminator Loss : {} / Generator Loss  : {} / {} loss : {}".format(Loss_D, Loss_G,
                                                                                               distance_loss,
                                                                                               Loss_Distance))
                else:
                    print(
                        "Discriminator Loss : {} / Generator Loss  : {}".format(Loss_D, Loss_G, distance_loss))

                if epoch % display_step == 0:
                    summary_str = sess.run(summary_operation, feed_dict=feed_dict_all)
                    summary_writer.add_summary(summary_str, global_step=epoch)

                    save_all_model_path = os.path.join(model_name, 'All/')
                    save_generator_model_path = os.path.join(model_name, 'Generator/')

                    if not os.path.exists(save_all_model_path):
                        os.makedirs(save_all_model_path)
                    if not os.path.exists(save_generator_model_path):
                        os.makedirs(save_generator_model_path)

                    saver_all.save(sess, save_all_model_path, global_step=epoch,
                                   write_meta_graph=False)
                    saver_generator.save(sess, save_generator_model_path,
                                         global_step=epoch,
                                         write_meta_graph=False)

            print("Optimization Finished!")

        if TEST:
            column_size = 10
            row_size = 10
            feed_dict = {z: np.random.normal(loc=0.0, scale=1.0, size=(column_size * row_size, noise_size)),
                         target: np.tile(np.diag(np.ones(column_size)), (row_size, 1))}

            generated_image = sess.run(G, feed_dict=feed_dict)
            show_image(model_name, generated_image, column_size=column_size, row_size=row_size)


if __name__ == "__main__":
    # optimizers_ selection = "Adam" or "RMSP" or "SGD"
    # 원하는 숫자를 생성이 가능하다.
    # 흑백 사진(Generator의 입력으로) -> 컬러(Discriminator의 입력)로 만들기
    # 선화(Generator의 입력)를 (여기서 채색된 선화는 Discriminator의 입력이 된다.)채색가능
    '''
    targeting = False 일 때는 숫자를 무작위로 생성하는 GAN MODEL 생성 - General GAN
    targeting = True 일 때는 숫자를 타게팅 하여 생성하는 GAN MODEL 생성 - Conditional GAN 

    targeting = True 일 때 -> distance_loss = 'L1' 일 경우 , generator에서 나오는 출력과 실제 출력값을 비교하는 L1 loss를 생성
    targeting = True 일 때 -> distance_loss = 'L2' 일 경우 , generator에서 나오는 출력과 실제 출력값을 비교하는 L2 loss를 생성
    targeting = True 일 때 -> distamce_loss = None 일 경우 , 추가적인 loss 없음
    '''
    # batch normalization은 Hidden Layer에만 추가합니다. 또한 활성화 함수전에 적용합니다.
    # regularization -> batch_norm = False 일때, L2 or L1 or nothing
    model(TEST=True, noise_size=128, targeting=True, distance_loss="L1",
          distance_loss_weight=1, \
          optimizer_selection="Adam", learning_rate=0.0002, training_epochs=50,
          batch_size=128,
          display_step=1, regularization='L2', scale=0.0001)

else:
    print("model imported")
