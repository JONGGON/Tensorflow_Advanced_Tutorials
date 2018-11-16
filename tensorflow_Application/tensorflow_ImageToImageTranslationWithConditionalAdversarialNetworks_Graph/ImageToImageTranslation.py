import shutil

from Dataset import *


def visualize(model_name="Pix2PixConditionalGAN", named_images=None, save_path=None):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # 이미지 y축 방향으로 붙이기
    image = np.hstack(named_images[1:])
    # 이미지 스케일 바꾸기(~1 ~ 1 -> 0~ 255)
    image = ((image + 1) * 127.5).astype(np.uint8)
    # RGB로 바꾸기
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(save_path, '{}_{}.png'.format(model_name, named_images[0])), image)
    print("<<< {}_{}.png saved in {} folder >>>".format(model_name, named_images[0], save_path))


def model(DB_name="facades",
          TEST=False,
          TFRecord=True,
          AtoB=False,
          Inputsize_limit=(256, 256),
          filter_size=32,
          norm_selection="BN",
          regularizer="L1",
          scale=0.0001,
          Dropout_rate=0.5,
          distance_loss="L1",
          distance_loss_weight=100,
          optimizer_selection="Adam",
          beta1=0.5, beta2=0.999,
          decay=0.999, momentum=0.9,
          image_pool=False,
          image_pool_size=50,
          learning_rate=0.0002, training_epochs=2, batch_size=2, display_step=1,
          inference_size=(512, 512),
          using_moving_variable=False,
          only_draw_graph=False,
          show_translated_image=True,
          weights_to_numpy=False,
          save_path="translated_image"):
    model_name = str(filter_size)

    if AtoB:
        model_name += "AtoB"
    else:
        model_name += "BtoA"
    model_name += DB_name

    if distance_loss == "L1":
        print("<<< target generative GAN with L1 loss >>>")
        model_name += "L1"
    elif distance_loss == "L2":
        print("<<< target generative GAN with L2 loss >>>")
        model_name += "L2"
    else:
        print("<<< target generative GAN >>>")

    if norm_selection == "BN":
        model_name = model_name + "BN"
    elif norm_selection == "IN":
        model_name = model_name + "IN"

    if batch_size == 1 and norm_selection == "BN":
        norm_selection = "IN"
        model_name = model_name[:-2] + "IN"

    if regularizer == "L1" or regularizer =="L2":
        model_name =  model_name + "reg" + regularizer

    if TEST == False:
        if os.path.exists("tensorboard/{}".format(model_name)):
            shutil.rmtree("tensorboard/{}".format(model_name))

    # stride? -> [1, 2, 2, 1] = [one image, width, height, one channel]
    def conv2d(input, weight_shape=None, bias_shape=None, norm_selection=None,
               strides=[1, 1, 1, 1], padding="VALID"):

        # weight_init = tf.contrib.layers.xavier_initializer(uniform=False)
        weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
        bias_init = tf.constant_initializer(value=0)

        weight_decay = tf.constant(scale, dtype=tf.float32)
        if regularizer == "L1":
            w = tf.get_variable("w", weight_shape, initializer=weight_init,
                                regularizer=tf.contrib.layers.l1_regularizer(scale=weight_decay))
        elif regularizer == "L2":
            w = tf.get_variable("w", weight_shape, initializer=weight_init,
                                regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay))
        else:
            w = tf.get_variable("w", weight_shape, initializer=weight_init)

        b = tf.get_variable("b", bias_shape, initializer=bias_init)
        conv_out = tf.nn.conv2d(input, w, strides=strides, padding=padding)

        # batch_norm을 적용하면 bias를 안써도 된다곤 하지만, 나는 썼다.
        if norm_selection == "BN":
            return tf.layers.batch_normalization(tf.nn.bias_add(conv_out, b), training=BN_FLAG)
        elif norm_selection == "IN":
            return tf.contrib.layers.instance_norm(tf.nn.bias_add(conv_out, b))
        else:
            return tf.nn.bias_add(conv_out, b)

    def conv2d_transpose(input, output_shape=None, weight_shape=None, bias_shape=None, norm_selection=None,
                         strides=[1, 1, 1, 1], padding="VALID"):

        weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
        bias_init = tf.constant_initializer(value=0)

        weight_decay = tf.constant(scale, dtype=tf.float32)
        if regularizer == "L1":
            w = tf.get_variable("w", weight_shape, initializer=weight_init,
                                regularizer=tf.contrib.layers.l1_regularizer(scale=weight_decay))
        elif regularizer == "L2":
            w = tf.get_variable("w", weight_shape, initializer=weight_init,
                                regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay))
        else:
            w = tf.get_variable("w", weight_shape, initializer=weight_init)

        b = tf.get_variable("b", bias_shape, initializer=bias_init)
        conv_out = tf.nn.conv2d_transpose(input, w, output_shape=output_shape, strides=strides, padding=padding)

        # batch_norm을 적용하면 bias를 안써도 된다곤 하지만, 나는 썼다.
        if norm_selection == "BN":
            return tf.layers.batch_normalization(tf.nn.bias_add(conv_out, b), training=BN_FLAG)
        elif norm_selection == "IN":
            return tf.contrib.layers.instance_norm(tf.nn.bias_add(conv_out, b))
        else:
            return tf.nn.bias_add(conv_out, b)

    # 유넷 - U-NET
    def generator(images=None):

        '''encoder의 활성화 함수는 모두 leaky_relu이며, decoder의 활성화 함수는 모두 relu이다.
        encoder의 첫번째 층에는 batch_norm이 적용 안된다.

        총 16개의 층이다.
        '''
        with tf.variable_scope("Generator"):
            with tf.variable_scope("encoder"):
                with tf.variable_scope("conv1"):
                    conv1 = conv2d(images, weight_shape=(4, 4, 3, filter_size), bias_shape=(filter_size),
                                   strides=[1, 2, 2, 1], padding="SAME")
                    # result shape = (batch_size, 128, 128, 64)
                with tf.variable_scope("conv2"):
                    conv2 = conv2d(tf.nn.leaky_relu(conv1, alpha=0.2),
                                   weight_shape=(4, 4, filter_size, filter_size * 2), bias_shape=(filter_size * 2),
                                   norm_selection=norm_selection,
                                   strides=[1, 2, 2, 1], padding="SAME")
                    # result shape = (batch_size, 64, 64, 128)
                with tf.variable_scope("conv3"):
                    conv3 = conv2d(tf.nn.leaky_relu(conv2, alpha=0.2),
                                   weight_shape=(4, 4, filter_size * 2, filter_size * 4), bias_shape=(filter_size * 4),
                                   norm_selection=norm_selection,
                                   strides=[1, 2, 2, 1], padding="SAME")
                    # result shape = (batch_size, 32, 32, 256)
                with tf.variable_scope("conv4"):
                    conv4 = conv2d(tf.nn.leaky_relu(conv3, alpha=0.2),
                                   weight_shape=(4, 4, filter_size * 4, filter_size * 8), bias_shape=(filter_size * 8),
                                   norm_selection=norm_selection,
                                   strides=[1, 2, 2, 1], padding="SAME")
                    # result shape = (batch_size, 16, 16, 512)
                with tf.variable_scope("conv5"):
                    conv5 = conv2d(tf.nn.leaky_relu(conv4, alpha=0.2),
                                   weight_shape=(4, 4, filter_size * 8, filter_size * 8), bias_shape=(filter_size * 8),
                                   norm_selection=norm_selection,
                                   strides=[1, 2, 2, 1], padding="SAME")
                    # result shape = (batch_size, 8, 8, 512)
                with tf.variable_scope("conv6"):
                    conv6 = conv2d(tf.nn.leaky_relu(conv5, alpha=0.2),
                                   weight_shape=(4, 4, filter_size * 8, filter_size * 8), bias_shape=(filter_size * 8),
                                   norm_selection=norm_selection,
                                   strides=[1, 2, 2, 1], padding="SAME")
                    # result shape = (batch_size, 4, 4, 512)
                with tf.variable_scope("conv7"):
                    conv7 = conv2d(tf.nn.leaky_relu(conv6, alpha=0.2),
                                   weight_shape=(4, 4, filter_size * 8, filter_size * 8), bias_shape=(filter_size * 8),
                                   norm_selection=norm_selection,
                                   strides=[1, 2, 2, 1], padding="SAME")
                    # result shape = (batch_size, 2, 2, 512)
                with tf.variable_scope("conv8"):
                    conv8 = conv2d(tf.nn.leaky_relu(conv7, alpha=0.2),
                                   weight_shape=(4, 4, filter_size * 8, filter_size * 8), bias_shape=(filter_size * 8),
                                   strides=[1, 2, 2, 1], padding="SAME")
                    # result shape = (batch_size, 1, 1, 512)

            with tf.variable_scope("decoder"):
                with tf.variable_scope("trans_conv1"):
                    '''output_shape = tf.shape(conv2) ???
                    output_shape 을 직접 지정 해주는 경우 예를 들어 (batch_size, 2, 2, 512) 이런식으로 지정해준다면,
                    trans_conv1 의 결과는 무조건 (batch_size, 2, 2, 512) 이어야 한다. 그러나 tf.shape(conv2)로 쓸 경우
                    나중에 session에서 실행될 때 입력이 되므로, batch_size에 종속되지 않는다. 
                    어쨌든 output_shape = tf.shape(conv2) 처럼 코딩하는게 무조건 좋다. 
                    '''
                    trans_conv1 = tf.nn.dropout(
                        conv2d_transpose(tf.nn.relu(conv8), output_shape=tf.shape(conv7),
                                         weight_shape=(4, 4, filter_size * 8, filter_size * 8),
                                         bias_shape=(filter_size * 8), norm_selection=norm_selection,
                                         strides=[1, 2, 2, 1], padding="SAME"), keep_prob=Dropout_rate)
                    # result shape = (batch_size, 2, 2, 512)
                    # 주의 : 활성화 함수 들어가기전의 encoder 요소를 concat 해줘야함
                    trans_conv1 = tf.concat([trans_conv1, conv7], axis=-1)
                    # result shape = (batch_size, 2, 2, 1024)

                with tf.variable_scope("trans_conv2"):
                    trans_conv2 = tf.nn.dropout(
                        conv2d_transpose(tf.nn.relu(trans_conv1), output_shape=tf.shape(conv6),
                                         weight_shape=(4, 4, filter_size * 8, filter_size * 16),
                                         bias_shape=(filter_size * 8), norm_selection=norm_selection,
                                         strides=[1, 2, 2, 1], padding="SAME"), keep_prob=Dropout_rate)
                    trans_conv2 = tf.concat([trans_conv2, conv6], axis=-1)
                    # result shape = (batch_size, 4, 4, 1024)

                with tf.variable_scope("trans_conv3"):
                    trans_conv3 = tf.nn.dropout(
                        conv2d_transpose(tf.nn.relu(trans_conv2), output_shape=tf.shape(conv5),
                                         weight_shape=(4, 4, filter_size * 8, filter_size * 16),
                                         bias_shape=(filter_size * 8), norm_selection=norm_selection,
                                         strides=[1, 2, 2, 1], padding="SAME"), keep_prob=Dropout_rate)
                    trans_conv3 = tf.concat([trans_conv3, conv5], axis=-1)
                    # result shape = (batch_size, 8, 8, 1024)

                with tf.variable_scope("trans_conv4"):
                    trans_conv4 = conv2d_transpose(tf.nn.relu(trans_conv3), output_shape=tf.shape(conv4),
                                                   weight_shape=(4, 4, filter_size * 8, filter_size * 16),
                                                   bias_shape=(filter_size * 8), norm_selection=norm_selection,
                                                   strides=[1, 2, 2, 1], padding="SAME")
                    trans_conv4 = tf.concat([trans_conv4, conv4], axis=-1)
                    # result shape = (batch_size, 16, 16, 1024)
                with tf.variable_scope("trans_conv5"):
                    trans_conv5 = conv2d_transpose(tf.nn.relu(trans_conv4), output_shape=tf.shape(conv3),
                                                   weight_shape=(4, 4, filter_size * 4, filter_size * 16),
                                                   bias_shape=(filter_size * 4), norm_selection=norm_selection,
                                                   strides=[1, 2, 2, 1], padding="SAME")
                    trans_conv5 = tf.concat([trans_conv5, conv3], axis=-1)
                    # result shape = (batch_size, 32, 32, 512)
                with tf.variable_scope("trans_conv6"):
                    trans_conv6 = conv2d_transpose(tf.nn.relu(trans_conv5), output_shape=tf.shape(conv2),
                                                   weight_shape=(4, 4, filter_size * 2, filter_size * 8),
                                                   bias_shape=(filter_size * 2), norm_selection=norm_selection,
                                                   strides=[1, 2, 2, 1], padding="SAME")
                    trans_conv6 = tf.concat([trans_conv6, conv2], axis=-1)
                    # result shape = (batch_size, 64, 64, 256)
                with tf.variable_scope("trans_conv7"):
                    trans_conv7 = conv2d_transpose(tf.nn.relu(trans_conv6), output_shape=tf.shape(conv1),
                                                   weight_shape=(4, 4, filter_size, filter_size * 4),
                                                   bias_shape=(filter_size), norm_selection=norm_selection,
                                                   strides=[1, 2, 2, 1], padding="SAME")
                    trans_conv7 = tf.concat([trans_conv7, conv1], axis=-1)
                    # result shape = (batch_size, 128, 128, 128)
                with tf.variable_scope("trans_conv8"):
                    output = tf.nn.tanh(
                        conv2d_transpose(tf.nn.relu(trans_conv7), output_shape=tf.shape(images),
                                         weight_shape=(4, 4, 3, filter_size * 2),
                                         bias_shape=(3),
                                         strides=[1, 2, 2, 1], padding="SAME"))
                    # result shape = (batch_size, 256, 256, 3)
        return output

    # PatchGAN
    def discriminator(images=None, condition=None):

        '''discriminator의 활성화 함수는 모두 leaky_relu이다.
        genertor와 마찬가지로 첫번째 층에는 batch_norm을 적용 안한다.

        왜 이런 구조를 사용? 아래의 구조 출력단의 ReceptiveField 크기를 구해보면 70이다.(ReceptiveFieldArithmetic/rf.py 에서 구해볼 수 있다.)'''
        conditional_input = tf.concat([images, condition], axis=-1)
        with tf.variable_scope("Discriminator"):
            with tf.variable_scope("conv1"):
                conv1 = tf.nn.leaky_relu(
                    conv2d(conditional_input, weight_shape=(4, 4, 6, filter_size),
                           bias_shape=(filter_size),
                           strides=[1, 2, 2, 1], padding="SAME"), alpha=0.2)
                # result shape = (batch_size, 128, 128, 64)
            with tf.variable_scope("conv2"):
                conv2 = tf.nn.leaky_relu(
                    conv2d(conv1, weight_shape=(4, 4, filter_size, filter_size * 2), bias_shape=(filter_size * 2),
                           norm_selection=norm_selection,
                           strides=[1, 2, 2, 1], padding="SAME"), alpha=0.2)
                # result shape = (batch_size, 64, 64, 128)
            with tf.variable_scope("conv3"):
                conv3 = conv2d(conv2, weight_shape=(4, 4, filter_size * 2, filter_size * 4),
                               bias_shape=(filter_size * 4), norm_selection=norm_selection,
                               strides=[1, 2, 2, 1], padding="SAME")
                # result shape = (batch_size, 32, 32, 256)
                conv3 = tf.nn.leaky_relu(
                    tf.pad(conv3, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT", constant_values=0), alpha=0.2)
                # result shape = (batch_size, 34, 34, 256)
            with tf.variable_scope("conv4"):
                conv4 = conv2d(conv3, weight_shape=(4, 4, filter_size * 4, filter_size * 8),
                               bias_shape=(filter_size * 8), norm_selection=norm_selection,
                               strides=[1, 1, 1, 1], padding="VALID")
                # result shape = (batch_size, 31, 31, 256)
                conv4 = tf.nn.leaky_relu(
                    tf.pad(conv4, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT", constant_values=0), alpha=0.2)
                # result shape = (batch_size, 33, 33, 512)
            with tf.variable_scope("output"):
                output = conv2d(conv4, weight_shape=(4, 4, filter_size * 8, 1), bias_shape=(1),
                                strides=[1, 1, 1, 1], padding="VALID")
                # result shape = (batch_size, 30, 30, 1)
            return output, tf.nn.sigmoid(output)

    def training(cost, var_list, scope=None):

        if regularizer=="L1" or regularizer=="L2":
            cost = tf.add_n([cost] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=scope))
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)):
            if optimizer_selection == "Adam":
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2)
            elif optimizer_selection == "RMSP":
                optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=decay, momentum=momentum)
            elif optimizer_selection == "SGD":
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            train_operation = optimizer.minimize(cost, var_list=var_list)
        return train_operation

    def min_max_loss(logits=None, labels=None):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

    if not TEST:
        # print(tf.get_default_graph()) #기본그래프이다.
        JG_Graph = tf.Graph()  # 내 그래프로 설정한다.- 혹시라도 나중에 여러 그래프를 사용할 경우를 대비
        with JG_Graph.as_default():  # as_default()는 JG_Graph를 기본그래프로 설정한다.

            with tf.name_scope("BN_FLAG"):
                if norm_selection == "BN":
                    BN_FLAG = tf.placeholder(tf.bool, shape=None)

            # 데이터 전처리
            with tf.name_scope("Dataset"):
                dataset = Dataset(DB_name=DB_name, AtoB=AtoB, batch_size=batch_size, use_TrainDataset=not TEST,
                                  TFRecord=TFRecord)
                iterator, next_batch, data_length = dataset.iterator()

                # 알고리즘
                x = next_batch[0]
                target = next_batch[1]

            with tf.variable_scope("shared_variables", reuse=tf.AUTO_REUSE) as scope:
                with tf.name_scope("Generator"):
                    G = generator(images=x)
                with tf.name_scope("Discriminator"):
                    D_real, sigmoid_D_real = discriminator(images=target, condition=x)
                    # scope.reuse_variables()
                    D_gene, sigmoid_D_gene = discriminator(images=G, condition=x)

            # 학습할 Discriminator 변수 지정
            var_D = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='shared_variables/Discriminator')
            # 학습할 Generator 변수 지정
            var_G = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='shared_variables/Generator')

            # optimizer의 매개변수들을 저장하고 싶지 않다면 여기에 선언해야한다.
            with tf.name_scope("saver"):
                saver_all = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=3)
                saver_generator = tf.train.Saver(var_list=var_G, max_to_keep=3)

            # Algorithjm - 속이고 속이는 과정

            with tf.name_scope("DiscriminatorLoss"):
                # for discriminator
                D_Loss = min_max_loss(logits=D_real, labels=tf.ones_like(D_real)) + min_max_loss(logits=D_gene,
                                                                                                 labels=tf.zeros_like(
                                                                                                     D_gene))

            with tf.name_scope("Generator_Loss"):
                # for generator
                G_Loss = min_max_loss(logits=D_gene, labels=tf.ones_like(D_gene))

            if distance_loss == "L1":
                with tf.name_scope("{}_loss".format(distance_loss)):
                    dis_loss = tf.losses.absolute_difference(target, G)
                    Gdis_Loss = G_Loss + tf.multiply(dis_loss, distance_loss_weight)
            elif distance_loss == "L2":
                with tf.name_scope("{}_loss".format(distance_loss)):
                    dis_loss = tf.losses.mean_squared_error(target, G)
                    Gdis_Loss = G_Loss + tf.multiply(dis_loss, distance_loss_weight)

            with tf.name_scope("Discriminator_trainer"):
                D_train_op = training(D_Loss, var_D, scope='shared_variables/Discriminator')
            with tf.name_scope("Generator_trainer"):
                if distance_loss == "L1" or distance_loss == "L2":
                    G_train_op = training(Gdis_Loss, var_G, scope='shared_variables/Generator')
                else:
                    G_train_op = training(G_Loss, var_G, scope='shared_variables/Generator')

            with tf.name_scope("Visualizer_each"):
                tf.summary.image("x", x, max_outputs=1)
                tf.summary.image("target", target, max_outputs=1)
                tf.summary.image("G", G, max_outputs=1)

            # tensorboard에 띄우기
            with tf.name_scope("Visualizer_stacked"):
                # 순서 : 입력, 타깃, 생성
                stacked_image = tf.concat([x, target, G], axis=2)
                tf.summary.image("stacked", stacked_image, max_outputs=3)

            with tf.name_scope("Loss"):
                tf.summary.scalar("DLoss", D_Loss)
                tf.summary.scalar("GLoss", G_Loss)
                if distance_loss == "L1" or distance_loss == "L2":
                    tf.summary.scalar("{}Loss".format(distance_loss), dis_loss)

            summary_operation = tf.summary.merge_all()

            '''
            WHY? 아래 6줄의 코드를 적어 주지 않고, 학습을 하게되면, TEST부분에서 tf.train.import_meta_graph를 사용할 때 오류가 발생한다. 
            -> 단순히 그래프를 가져오고 가중치를 복원하는 것만으로는 안된다. 세션을 실행할때 인수로 사용할 변수에 대한 
            추가 접근을 제공하지 않기 때문에 아래와 같이 저장을 해놓은 뒤 TEST 시에 불러와서 다시 사용 해야한다. - 그렇지 않으면, JG.get_operations() 함수를
            사용해 출력된 모든 연산의 리스트에서 하나하나 찾아야한다.
            필요한 변수가 있을 시 아래와 같이 추가해서 그래프를 새로 만들어 주면 된다.
            '''
            if norm_selection == "BN":
                for op in (x, G, BN_FLAG):
                    tf.add_to_collection("way", op)
            else:
                for op in (x, G):
                    tf.add_to_collection("way", op)

            # 아래와 같은 코드도 가능.
            # tf.add_to_collection('x', x)
            # tf.add_to_collection('G', G)
            # tf.add_to_collection('BN_FLAG', BN_FLAG)

            # generator graph 구조를 파일에 쓴다.
            meta_save_file_path = os.path.join(model_name, 'Generator', 'Generator_Graph.meta')
            saver_generator.export_meta_graph(meta_save_file_path, collection_list=["way"])

            if only_draw_graph:
                print('<<< Generator_Graph.meta 파일만 저장하고 종료합니다. >>>')
                exit(0)

            if image_pool and batch_size == 1:
                imagepool = ImagePool(image_pool_size=image_pool_size)

            config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            # config.gpu_options.per_process_gpu_memory_fraction = 0.1

            with tf.Session(graph=JG_Graph, config=config) as sess:
                print("<<< initializing!!! >>>")
                sess.run(tf.global_variables_initializer())
                ckpt_all = tf.train.get_checkpoint_state(os.path.join(model_name, 'All'))

                if (ckpt_all and tf.train.checkpoint_exists(ckpt_all.model_checkpoint_path)):
                    print("<<< all variable retored except for optimizer parameter >>>")
                    print("<<< Restore {} checkpoint!!! >>>".format(os.path.basename(ckpt_all.model_checkpoint_path)))
                    saver_all.restore(sess, ckpt_all.model_checkpoint_path)

                summary_writer = tf.summary.FileWriter(os.path.join("tensorboard", model_name), sess.graph)
                sess.run(iterator.initializer)
                for epoch in tqdm(range(1, training_epochs + 1)):

                    Loss_D = 0
                    Loss_G = 0
                    if distance_loss == "L1" or distance_loss == "L2":
                        Loss_Distance = 0
                    # 아래의 두 변수가 각각 0.5 씩의 값을 갖는게 가장 이상적이다.
                    sigmoid_D = 0
                    sigmoid_G = 0

                    total_batch = int(data_length / batch_size)
                    for i in range(total_batch):
                        # 입력 이미지가 Inputsize_limit[0] x Inputsize_limit[1] 이하이면, exit()
                        temp = sess.run(x)
                        if temp.shape[1] < Inputsize_limit[0] or temp.shape[2] < Inputsize_limit[1]:
                            print("<<< 입력된 이미지 크기는 {} x {} 입니다. >>>".format(temp.shape[1], temp.shape[2]))
                            print("<<< 입력되는 이미지 크기는 {} x {} 보다 크거나 같아야 합니다. >>>".format(Inputsize_limit[0],
                                                                                        Inputsize_limit[1]))
                            print("<<< 강제 종료 합니다. >>>")
                            exit(0)
                        if norm_selection == "BN":
                            if distance_loss == "L1" or distance_loss == "L2":
                                _, Generator_Loss, Distance_Loss, D_gene_simgoid = sess.run(
                                    [G_train_op, G_Loss, dis_loss, sigmoid_D_gene], feed_dict={BN_FLAG: True})
                            else:
                                _, Generator_Loss, D_gene_simgoid = sess.run(
                                    [G_train_op, G_Loss, sigmoid_D_gene], feed_dict={BN_FLAG: True})
                        else:
                            if distance_loss == "L1" or distance_loss == "L2":
                                _, Generator_Loss, Distance_Loss, D_gene_simgoid = sess.run(
                                    [G_train_op, G_Loss, dis_loss, sigmoid_D_gene])
                            else:
                                _, Generator_Loss, D_gene_simgoid = sess.run(
                                    [G_train_op, G_Loss, sigmoid_D_gene])

                        # image_pool 변수 사용할 때(단 batch_size=1 일 경우만), Discriminator Update
                        if image_pool and batch_size == 1:
                            fake_G = imagepool(image=sess.run(G))
                            # G 에 과거에 생성된 fake_G를 넣어주자!!!
                            _, Discriminator_Loss, D_real_simgoid = sess.run([D_train_op, D_Loss, sigmoid_D_real],
                                                                             feed_dict={G: fake_G})
                        # image_pool 변수를 사용하지 않을 때, Discriminator Update
                        else:
                            if norm_selection == "BN":
                                _, Discriminator_Loss, D_real_simgoid = sess.run([D_train_op, D_Loss, sigmoid_D_real],
                                                                                 feed_dict={BN_FLAG: True})
                            else:
                                _, Discriminator_Loss, D_real_simgoid = sess.run([D_train_op, D_Loss, sigmoid_D_real])

                        Loss_D += (Discriminator_Loss / total_batch)
                        Loss_G += (Generator_Loss / total_batch)
                        if distance_loss == "L1" or distance_loss == "L2":
                            Loss_Distance += (Distance_Loss / total_batch)
                        sigmoid_D += np.mean(D_real_simgoid, axis=(1, 2, 3)) / total_batch
                        sigmoid_G += np.mean(D_gene_simgoid, axis=(1, 2, 3)) / total_batch
                        print("<<< {} epoch : {} batch running of {} total batch... >>>".format(epoch, i, total_batch))

                    print(
                        "<<< Discriminator mean output : {} / Generator mean output : {} >>>".format(np.mean(sigmoid_D),
                                                                                                     np.mean(
                                                                                                         sigmoid_G)))

                    if distance_loss == "L1" or distance_loss == "L2":
                        print(
                            "<<< Discriminator Loss : {} / Generator Loss  : {} / {} loss : {} >>>".format(Loss_D,
                                                                                                           Loss_G,
                                                                                                           distance_loss,
                                                                                                           Loss_Distance))
                    else:
                        print(
                            "<<< Discriminator Loss : {} / Generator Loss  : {} >>>".format(Loss_D, Loss_G))

                    if epoch % display_step == 0:
                        if norm_selection == "BN":
                            summary_str = sess.run(summary_operation, feed_dict={BN_FLAG: True})
                        else:
                            summary_str = sess.run(summary_operation)

                        summary_writer.add_summary(summary_str, global_step=epoch)

                        save_all_model_path = os.path.join(model_name, 'All')
                        save_generator_model_path = os.path.join(model_name, 'Generator')

                        if not os.path.exists(save_all_model_path):
                            os.makedirs(save_all_model_path)
                        if not os.path.exists(save_generator_model_path):
                            os.makedirs(save_generator_model_path)

                        saver_all.save(sess, save_all_model_path + "/", global_step=epoch,
                                       write_meta_graph=False)
                        saver_generator.save(sess, save_generator_model_path + "/", global_step=epoch,
                                             write_meta_graph=False)

                print("<<< Optimization Finished! >>>")

    else:
        tf.reset_default_graph()
        meta_path = glob.glob(os.path.join(model_name, 'Generator', '*.meta'))
        if len(meta_path) == 0:
            print("<<< Graph가 존재 하지 않습니다. >>>")
            exit(0)
        else:
            print("<<< Graph가 존재 합니다. >>>")

        # print(tf.get_default_graph()) #기본그래프이다.
        JG = tf.Graph()  # 내 그래프로 설정한다.- 혹시라도 나중에 여러 그래프를 사용할 경우를 대비
        with JG.as_default():  # as_default()는 JG를 기본그래프로 설정한다.

            saver = tf.train.import_meta_graph(meta_path[0], clear_devices=True)  # meta graph 읽어오기
            if saver == None:
                print("<<< meta 파일을 읽을 수 없습니다. >>>")
                exit(0)

            '''
            WHY? 아래 4줄의 코드를 적어 주지 않으면 오류가 발생한다. 
            -> 단순히 그래프를 가져오고 가중치를 복원하는 것만으로는 안된다. 세션을 실행할때 인수로 사용할 변수에 대한 
            추가 접근을 제공하지 않기 때문에 아래와 같이 get_colltection으로 입,출력 변수들을 불러와서 다시 사용 해야 한다.
            '''
            if norm_selection == "BN":
                x, G, BN_FLAG = tf.get_collection('way')
            else:
                x, G = tf.get_collection('way')

            # Test Dataset 가져오기
            dataset = Dataset(DB_name=DB_name, AtoB=AtoB, use_TrainDataset=not TEST,
                              inference_size=inference_size, TFRecord=TFRecord)
            iterator, next_batch, data_length = dataset.iterator()

            with tf.Session(graph=JG) as sess:
                sess.run(iterator.initializer)
                ckpt = tf.train.get_checkpoint_state(os.path.join(model_name, 'Generator'))

                if ckpt == None:
                    print("<<< checkpoint file does not exist>>>")
                    print("<<< Exit the program >>>")
                    exit(0)

                if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                    print("<<< generator variable retored except for optimizer parameter >>>")
                    print("<<< Restore {} checkpoint!!! >>>".format(os.path.basename(ckpt.model_checkpoint_path)))
                    saver.restore(sess, ckpt.model_checkpoint_path)

                # Generator에서 생성된 이미지 저장
                if show_translated_image:
                    for i in range(data_length):
                        x_numpy, target_numpy = sess.run(next_batch)
                        # 입력 이미지가 Inputsize_limit[0] xInputsize_limit[1] 이하이면, exit()
                        if x_numpy.shape[1] < Inputsize_limit[0] or x_numpy.shape[2] < Inputsize_limit[1]:
                            print("<<< 입력된 이미지 크기는 {} x {} 입니다. >>>".format(x_numpy.shape[1], x_numpy.shape[2]))
                            print("<<< 입력되는 이미지 크기는 {} x {} 보다 크거나 같아야 합니다. >>>".format(Inputsize_limit[0],
                                                                                        Inputsize_limit[1]))
                            print("<<< 강제 종료 합니다. >>>")
                            exit(0)

                        if norm_selection == "BN":
                            translated_image = sess.run(G, feed_dict={x: x_numpy, BN_FLAG: not using_moving_variable})
                        else:
                            translated_image = sess.run(G, feed_dict={x: x_numpy})

                        # 순서 : 입력, 타깃, 생성
                        visualize(model_name=model_name,
                                  named_images=[i, x_numpy[0], target_numpy[0], translated_image[0]],
                                  save_path=save_path)

                # 가중치 저장 - 약간 생소한 API지만 유용.
                if weights_to_numpy:
                    numpy_weight_save_path = "NumpyWeightOfModel"
                    if not os.path.exists(numpy_weight_save_path):
                        os.makedirs(numpy_weight_save_path)
                    # 1, checkpoint 읽어오는 것
                    reader = tf.train.NewCheckpointReader(ckpt.model_checkpoint_path)
                    dtype = list(reader.get_variable_to_dtype_map().values())[0]
                    ''' 2. tf.train.NewCheckpointReader에도
                    reader.get_variable_to_dtype_map() -> 이름 , dype 반환 or reader.get_variable_to_shape_map() 이름 , 형태 반환 
                    하는데 사전형이라 순서가 중구난방이다.
                    요 아래의 것은 리스트 형태로 name, shape이 순서대로 나온다.
                    '''
                    name_shape = tf.contrib.framework.list_variables(ckpt.model_checkpoint_path)
                    with open(os.path.join(numpy_weight_save_path, "name_shape_info.txt"), mode='w') as f:
                        f.write("                      < weight 정보 >\n\n")
                        f.write("파일 개수 : {}개\n\n".format(len(name_shape)))
                        f.write("------------------- 1. data type ---------------------\n\n")
                        f.write("{} \n\n".format(str(dtype).strip("<>").replace(":", " :")))
                        print("------------------------------------------------------")
                        print("<<< 총 파일 개수 : {} >>>".format(len(name_shape)))

                        f.write("-------------- 2. weight name, shape ----------------\n\n")
                        for name, shape in name_shape:
                            # 앞의 shared_variables / Generator 빼버리기
                            seperated = name.split("/")[2:]
                            joined = "_".join(seperated)
                            shape = str(shape).strip('[]')
                            print("##################################################")
                            print("<<< weight : {}.npy >>>".format(joined))
                            print("shape : ({})".format(shape))
                            f.write("<<< {}.npy >>>\n<<< shape : ({}) >>>\n\n".format(joined, shape))

                            # weight npy로 저장하기
                            np.save(os.path.join(numpy_weight_save_path, joined), reader.get_tensor(name))


if __name__ == "__main__":
    '''
    DB_name 은 아래에서 하나 고르자
    1. "cityscapes"
    2. "facades"
    3. "maps"
    AtoB -> A : image,  B : segmentation
    AtoB = True  -> image -> segmentation
    AtoB = False -> segmentation -> image
    '''
    # 256x256 크기 이상의 다양한 크기의 이미지를 동시 학습 하는 것이 가능하다.(256 X 256으로 크기 제한을 뒀다.)
    # -> 단 batch_size =  1 일 때만 가능하다. - batch_size>=2 일때 여러사이즈의 이미지를 동시에 학습 하고 싶다면, 각각 따로 사이즈별로 Dataset을 생성 후 학습시키면 된다.
    # pix2pix GAN이나, Cycle gan이나 데이터셋 자체가 같은 크기의 이미지를 다루므로, 위 설명을 무시해도 된다.
    # TEST=False 시 입력 이미지의 크기가 256x256 미만이면 강제 종료한다.
    # TEST=True 시 입력 이미지의 크기가 256x256 미만이면 강제 종료한다.
    # optimizers_ selection = "Adam" or "RMSP" or "SGD"
    model(DB_name="facades",
          TEST=False,  # TEST=False -> Training or TEST=True -> TEST
          # 대량의 데이터일 경우 TFRecord=True가 더 빠르다.
          TFRecord=True,  # TFRecord=True -> TFRecord파일로 저장한후 사용하는 방식 사용 or TFRecord=False -> 파일에서 읽어오는 방식 사용
          AtoB=False,  # 데이터 순서 변경(ex) AtoB=True : image -> segmentation / AtoB=False : segmetation -> image)
          Inputsize_limit=(256, 256),  # 입력되어야 하는 최소 사이즈를 내가 지정 - (256,256) 으로 하자
          filter_size=32,  # generator와 discriminator의 처음 layer의 filter 크기
          norm_selection="BN",  # IN - instance normalizaiton , BN -> batch normalization, NOTHING
          regularizer=" ",  # L1 or L2 정규화 -> 오버피팅 막기 위함
          scale=0.0001,  # L1 or L2 정규화 weight
          Dropout_rate=0.5,  # generator의 Dropout 비율
          distance_loss=" ",  # L2 or NOTHING
          distance_loss_weight=100,  # distance_loss의 가중치
          optimizer_selection="Adam",  # optimizers_ selection = "Adam" or "RMSP" or "SGD"
          beta1=0.5, beta2=0.999,  # for Adam optimizer
          decay=0.999, momentum=0.9,  # for RMSProp optimizer
          image_pool=False,  # discriminator 업데이트시 이전에 generator로 부터 생성된 이미지의 사용 여부
          image_pool_size=50,  # image_pool=True 라면 몇개를 사용 할지?
          learning_rate=0.0002, training_epochs=1, batch_size=1, display_step=1,
          inference_size=(256, 256),  # TEST=True 일때, inference 할 수 있는 최소의 크기를 256 x 256으로 크기 제한을 뒀다.
          using_moving_variable=False,  # TEST=True 일때, Moving Average를 Inference에 사용할지 말지 결정하는 변수
          only_draw_graph=False,  # TEST=False 일 때 only_draw_graph=True이면 그래프만 그리고 종료한다.
          show_translated_image=True,  # TEST=True 일 때 변환된 이미지를 보여줄지 말지
          weights_to_numpy=False,  # TEST=True 일 때 가중치를 npy 파일로 저장할지 말지
          save_path="translated_image")  # TEST=True 일 때 변환된 이미지가 저장될 폴더

else:
    print("model imported")
