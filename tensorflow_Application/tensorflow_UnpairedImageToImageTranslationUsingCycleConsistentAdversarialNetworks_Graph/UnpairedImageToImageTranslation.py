import shutil

from Dataset import *


def visualize(model_name="CycleGAN", named_images=None, save_path=None):
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


def model(
        DB_name="horse2zebra",
        TEST=False,
        TFRecord=True,
        Inputsize_limit=(256, 256),
        filter_size=8,
        norm_selection="BN",
        regularizer=" ",
        scale=0.0001,
        cycle_consistency_loss="L1",
        cycle_consistency_loss_weight=10,
        optimizer_selection="Adam",
        beta1=0.5, beta2=0.999,
        decay=0.999, momentum=0.9,
        use_identity_mapping=False,
        image_pool=True,
        image_pool_size=50,
        learning_rate=0.0002, training_epochs=1, batch_size=1, display_step=1,
        weight_decay_epoch=100,
        learning_rate_decay=0.99,
        inference_size=(256, 256),
        using_moving_variable=False,
        only_draw_graph=False,
        show_translated_image=True,
        save_path="translated_image",
        weights_to_numpy=False):
    print("<<< CycleGAN >>>")

    model_name = str(filter_size)
    model_name += DB_name

    if cycle_consistency_loss == "L1":
        model_name += "L1"  # L1 cycle consistency loss
    else:
        model_name += "L2"  # L1 cycle consistency loss

    if use_identity_mapping:
        model_name += "IML"  # identity mapping loss

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

    def residual_block(x):

        y = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        y = tf.nn.relu(conv2d(y, weight_shape=(3, 3, filter_size * 4, filter_size * 4), bias_shape=(filter_size * 4),
                              norm_selection=norm_selection,
                              strides=[1, 1, 1, 1], padding="VALID"))
        y = tf.pad(y, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        y = tf.nn.relu(conv2d(y, weight_shape=(3, 3, filter_size * 4, filter_size * 4), bias_shape=(filter_size * 4),
                              norm_selection=norm_selection,
                              strides=[1, 1, 1, 1], padding="VALID"))
        return y + x

    # Residual Net
    def generator(images=None, name=None):
        '''
        논문에서
        256x256 입력일 때 9 block!!!
        -layer 구성-
        c7s1-32 -> 7x7 Convolution-InstanceNorm-Relu-32filter-1stride
         |
        d64 -> 3x3 Convolution-InstanceNorm-Relu--64filter-2stride
         |
        d128 -> 3x3 Convolution-InstanceNorm-Relu--128filter-2stride
        |
        R128 -> P128 - R128 - R128 - R128 - R128 - R128 - R128 - R128 :  / 이게 9개
         |
        u64 -> 3x3 fractionalstridedConvolution-InstanceNorm-Relu-64filter-1/2stride
         |
        u32 -> 3x3 fractionalstridedConvolution-InstanceNorm-Relu-32filter-1/2stride
         |
        c7s1-3 -> 7x7 Convolution-InstanceNorm-Relu-3filter-1stride
        '''
        with tf.variable_scope(name):
            with tf.variable_scope("conv1"):
                padded_images = tf.pad(images, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
                conv1 = tf.nn.relu(
                    conv2d(padded_images, weight_shape=(7, 7, 3, filter_size), bias_shape=(filter_size),
                           norm_selection=norm_selection,
                           strides=[1, 1, 1, 1], padding="VALID"))
                # result shape = (batch_size, 256, 256, 32)
            with tf.variable_scope("conv2"):
                conv2 = tf.nn.relu(
                    conv2d(conv1, weight_shape=(3, 3, filter_size, filter_size * 2), bias_shape=(filter_size * 2),
                           norm_selection=norm_selection,
                           strides=[1, 2, 2, 1], padding="SAME"))
                # result shape = (batch_size, 128, 128, 64)
            with tf.variable_scope("conv3"):
                conv3 = tf.nn.relu(
                    conv2d(conv2, weight_shape=(3, 3, filter_size * 2, filter_size * 4), bias_shape=(filter_size * 4),
                           norm_selection=norm_selection,
                           strides=[1, 2, 2, 1], padding="SAME"))
                # result shape = (batch_size, 64, 64, 128)

            with tf.variable_scope("9_residual_block"):
                r1 = residual_block(conv3)
                r2 = residual_block(r1)
                r3 = residual_block(r2)
                r4 = residual_block(r3)
                r5 = residual_block(r4)
                r6 = residual_block(r5)
                r7 = residual_block(r6)
                r8 = residual_block(r7)
                r9 = residual_block(r8)
                # result shape = (batch_size, 64, 64, 128)

            with tf.variable_scope("trans_conv1"):
                '''output_shape = tf.shape(conv2) ???
                output_shape 을 직접 지정 해주는 경우 예를 들어 (batch_size, 2, 2, 512) 이런식으로 지정해준다면,
                trans_conv1 의 결과는 무조건 (batch_size, 2, 2, 512) 이어야 한다. 그러나 tf.shape(conv2)로 쓸 경우
                나중에 session에서 실행될 때 입력이 되므로, batch_size에 종속되지 않는다. 
                어쨌든 output_shape = tf.shape(conv2) 처럼 코딩하는게 무조건 좋다. 
                '''
                trans_conv1 = tf.nn.relu(conv2d_transpose(r9, output_shape=tf.shape(conv2),
                                                          weight_shape=(3, 3, filter_size * 2, filter_size * 4),
                                                          bias_shape=(filter_size * 2), norm_selection=norm_selection,
                                                          strides=[1, 2, 2, 1], padding="SAME"))
                # result shape = (batch_size, 128, 128, 64)

            with tf.variable_scope("trans_conv2"):
                trans_conv2 = tf.nn.relu(conv2d_transpose(trans_conv1, output_shape=tf.shape(conv1),
                                                          weight_shape=(3, 3, filter_size, filter_size * 2),
                                                          bias_shape=(filter_size), norm_selection=norm_selection,
                                                          strides=[1, 2, 2, 1], padding="SAME"))
                # result shape = (batch_size, 256, 256, 32)

            with tf.variable_scope("output"):
                padded_trans_conv2 = tf.pad(trans_conv2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
                output = tf.nn.tanh(
                    conv2d(padded_trans_conv2, weight_shape=(7, 7, filter_size, 3), bias_shape=(3),
                           strides=[1, 1, 1, 1],
                           padding="VALID"))
            # result shape = (batch_size, 256, 256, 3)
        return output

    # PatchGAN
    def discriminator(images=None, name=None):

        '''discriminator의 활성화 함수는 모두 leaky_relu(slope = 0.2)이다.
        첫 번째 층에는 instance normalization 을 적용하지 않는다.
        왜 이런 구조를 사용? 아래의 구조 출력단의 ReceptiveField 크기를 구해보면 70이다.(ReceptiveFieldArithmetic/rf.py 에서 구해볼 수 있다.)
        layer 구성 은 pix2pix gan 의 discriminator와 같다.(PatchGAN 70X70)
        '''
        with tf.variable_scope(name):
            with tf.variable_scope("conv1"):
                conv1 = tf.nn.leaky_relu(
                    conv2d(images, weight_shape=(4, 4, 3, filter_size * 2), bias_shape=(filter_size * 2),
                           strides=[1, 2, 2, 1], padding="SAME"), alpha=0.2)
                # result shape = (batch_size, 128, 128, 64)
            with tf.variable_scope("conv2"):
                conv2 = tf.nn.leaky_relu(
                    conv2d(conv1, weight_shape=(4, 4, filter_size * 2, filter_size * 4), bias_shape=(filter_size * 4),
                           norm_selection=norm_selection,
                           strides=[1, 2, 2, 1], padding="SAME"), alpha=0.2)
                # result shape = (batch_size, 64, 64, 128)
            with tf.variable_scope("conv3"):
                conv3 = conv2d(conv2, weight_shape=(4, 4, filter_size * 4, filter_size * 8),
                               bias_shape=(filter_size * 8), norm_selection=norm_selection,
                               strides=[1, 2, 2, 1], padding="SAME")
                # result shape = (batch_size, 32, 32, 256)
                conv3 = tf.nn.leaky_relu(
                    tf.pad(conv3, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT", constant_values=0), alpha=0.2)
                # result shape = (batch_size, 34, 34, 256)
            with tf.variable_scope("conv4"):
                conv4 = conv2d(conv3, weight_shape=(4, 4, filter_size * 8, filter_size * 16),
                               bias_shape=(filter_size * 16), norm_selection=norm_selection,
                               strides=[1, 1, 1, 1], padding="VALID")
                # result shape = (batch_size, 31, 31, 256)
                conv4 = tf.nn.leaky_relu(
                    tf.pad(conv4, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT", constant_values=0), alpha=0.2)
                # result shape = (batch_size, 33, 33, 512)
            with tf.variable_scope("output"):
                output = conv2d(conv4, weight_shape=(4, 4, filter_size * 16, 1), bias_shape=(1),
                                strides=[1, 1, 1, 1], padding="VALID")
                # result shape = (batch_size, 30, 30, 1)
            return tf.nn.sigmoid(output)

    def training(cost, var_list, scope=None):

        if regularizer=="L1" or regularizer=="L2":
            cost = tf.add_n([cost] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=scope))

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)):
            if optimizer_selection == "Adam":
                optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1, beta2=beta2)
            elif optimizer_selection == "RMSP":
                optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, decay=decay, momentum=momentum)
            elif optimizer_selection == "SGD":
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
            train_operation = optimizer.minimize(cost, var_list=var_list)
        return train_operation

    if not TEST:

        # print(tf.get_default_graph()) #기본그래프이다.
        JG_Graph = tf.Graph()  # 내 그래프로 설정한다.- 혹시라도 나중에 여러 그래프를 사용할 경우를 대비
        with JG_Graph.as_default():  # as_default()는 JG_Graph를 기본그래프로 설정한다.

            with tf.name_scope("BN_FLAG"):
                if norm_selection == "BN":
                    BN_FLAG = tf.placeholder(tf.bool, shape=None)
            '''
            학습률 - 논문에서 100epoch 후에 선형적으로 줄인다고 했으므로, 아래의 변수가 필요하다. 2가지 방법

            * Variable과 placeholder의 차이점
            tf.Variable 을 사용하면 선언 할 때 초기 값을 제공해야하고 
            tf.placeholder 를 사용하면 초기 값을 제공 할 필요가 없으므로 Session.run 의 feed_dict 인수를 사용하여 런타임에 지정할 수 있다.
            
            * Variable과 placeholder의 공통점
            둘 다 feed_dict 인수를 사용하여 런타임에 지정할 수 있다.
            
            * 추가설명
            궁금한 사람을 위해? Variable 도 feed_dict으로 값을 넣어 줄 수 있는 이유? 
            - 텐서플로는 그래프 구조이다. 이말은 그래프가 그려지고 나면, feed_dict 인수로 자유롭게 변수들에 접근이 가능하다는 말.
            '''

            # `GraphKeys.TRAINABLE_VARIABLES` 에 추가될 필요가 없으니 trainable=False 로 한다.
            with tf.name_scope("Learning_rate"):
                lr = tf.Variable(initial_value=learning_rate, trainable=False, dtype=tf.float32)

            # 일반적인 방법
            # lr = tf.placeholder(dtype=tf.float32)
            # 데이터 전처리
            with tf.name_scope("Dataset"):
                dataset = Dataset(DB_name=DB_name, batch_size=batch_size, TFRecord=TFRecord,
                                  use_TrainDataset=not TEST)
                A_iterator, A_next_batch, A_length, B_iterator, B_next_batch, B_length = dataset.iterator()

                # 알고리즘
                A = A_next_batch
                B = B_next_batch

            with tf.variable_scope("shared_variables", reuse=tf.AUTO_REUSE) as scope:

                with tf.name_scope("AtoB_Generator"):
                    AtoB_gene = generator(images=A, name="AtoB_generator")

                with tf.name_scope("BtoA_generator"):
                    BtoA_gene = generator(images=B, name="BtoA_generator")

                # A -> B -> A
                with tf.name_scope("Back_to_A"):
                    BackA = generator(images=AtoB_gene, name="BtoA_generator")

                # B -> A -> B
                with tf.name_scope("Back_to_B"):
                    BackB = generator(images=BtoA_gene, name="AtoB_generator")

                with tf.name_scope("AtoB_Discriminator"):
                    AtoB_Dreal = discriminator(images=B, name="AtoB_Discriminator")
                    # scope.reuse_variables()
                    AtoB_Dgene = discriminator(images=AtoB_gene, name="AtoB_Discriminator")

                with tf.name_scope("BtoA_Discriminator"):
                    BtoA_Dreal = discriminator(images=A, name="BtoA_Discriminator")
                    # scope.reuse_variables()
                    BtoA_Dgene = discriminator(images=BtoA_gene, name="BtoA_Discriminator")

                if use_identity_mapping:
                    with tf.name_scope("identity_generator"):
                        im_AtoB_GeneratorWithB = generator(images=B, name="AtoB_generator")
                        im_BtoA_GeneratorWithA = generator(images=A, name="BtoA_generator")

            with tf.name_scope("visualizer_each"):
                tf.summary.image("A", A, max_outputs=1)
                tf.summary.image("AtoB_gene", AtoB_gene, max_outputs=1)
                tf.summary.image("B", B, max_outputs=1)
                tf.summary.image("BtoA_gene", BtoA_gene, max_outputs=1)

            with tf.name_scope("visualizer_stacked"):
                stacked = tf.concat([A, AtoB_gene, B, BtoA_gene], axis=2)
                # 순서 -> Origin_Aimage, Origin_Bimage, AtoBImage, BtoAImage
                tf.summary.image("stacked", stacked, max_outputs=3)

            # 학습할 AtoB-Discriminator 변수 지정
            AtoB_varD = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='shared_variables/AtoB_Discriminator')

            # 학습할 AtoB-Generator 변수 지정
            AtoB_varG = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='shared_variables/AtoB_generator')

            # 학습할 BtoA-Discriminator 변수 지정
            BtoA_varD = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='shared_variables/BtoA_Discriminator')

            # 학습할 BtoA-Generator 변수 지정
            BtoA_varG = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='shared_variables/BtoA_generator')

            # optimizer의 매개변수들을 저장하고 싶지 않다면 여기에 선언해야한다.
            with tf.name_scope("saver"):
                saver_all = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=3)
                saver_generator = tf.train.Saver(var_list=AtoB_varG + BtoA_varG, max_to_keep=3)
            '''
            논문에 나와있듯이, log likelihood objective 대신 least-square loss를 사용한다.
            '''
            with tf.name_scope("AtoB_Discriminator_loss"):
                # for AtoB discriminator
                AtoB_DLoss = tf.reduce_mean(tf.square(AtoB_Dreal - tf.ones_like(AtoB_Dreal))) + tf.reduce_mean(
                    tf.square(AtoB_Dgene - tf.zeros_like(AtoB_Dgene)))

            with tf.name_scope("AtoB_Generator_loss"):
                # for AtoB generator
                AtoB_GLoss = tf.reduce_mean(tf.square(AtoB_Dgene - tf.ones_like(AtoB_Dgene)))

            with tf.name_scope("BtoA_Discriminator_loss"):
                # for BtoA discriminator
                BtoA_DLoss = tf.reduce_mean(tf.square(BtoA_Dreal - tf.ones_like(BtoA_Dreal))) + tf.reduce_mean(
                    tf.square(BtoA_Dgene - tf.zeros_like(BtoA_Dgene)))

            with tf.name_scope("BtoA_Generator_loss"):
                # for BtoA generator
                BtoA_GLoss = tf.reduce_mean(tf.square(BtoA_Dgene - tf.ones_like(BtoA_Dgene)))

            # Cycle Consistency Loss
            if cycle_consistency_loss == "L1":
                with tf.name_scope("{}_loss".format(cycle_consistency_loss)):
                    cycle_loss = tf.losses.absolute_difference(A, BackA) + tf.losses.absolute_difference(B, BackB)
                    AtoB_GLoss += tf.multiply(cycle_loss, cycle_consistency_loss_weight)
                    BtoA_GLoss += tf.multiply(cycle_loss, cycle_consistency_loss_weight)
            else:  # cycle_consistency_loss == "L2"
                with tf.name_scope("{}_loss".format(cycle_consistency_loss)):
                    cycle_loss = tf.losses.mean_squared_error(BackA, A) + tf.losses.mean_squared_error(BackB, B)
                    AtoB_GLoss += tf.multiply(cycle_loss, cycle_consistency_loss_weight)
                    BtoA_GLoss += tf.multiply(cycle_loss, cycle_consistency_loss_weight)

            # Identity mapping -> input과 output의 컬러 구성을 보존하기 위해 쓴다고 함(논문에서는 painting -> photo DB로 학습할 때 씀)
            if use_identity_mapping:
                with tf.name_scope("{}_loss".format("Identity_mapping_Loss")):
                    Identity_mapping_Loss = tf.losses.absolute_difference(im_AtoB_GeneratorWithB, B) + \
                                            tf.losses.absolute_difference(im_BtoA_GeneratorWithA, A)
                    AtoB_GLoss += tf.multiply(Identity_mapping_Loss, 0.5 * cycle_consistency_loss_weight)
                    BtoA_GLoss += tf.multiply(Identity_mapping_Loss, 0.5 * cycle_consistency_loss_weight)

            with tf.name_scope("AtoB_Discriminator_trainer"):
                AtoB_D_train_op = training(AtoB_DLoss, AtoB_varD, scope='shared_variables/AtoB_Discriminator')
            with tf.name_scope("AtoB_Generator_trainer"):
                AtoB_G_train_op = training(AtoB_GLoss, AtoB_varG, scope='shared_variables/AtoB_generator')
            with tf.name_scope("BtoA_Discriminator_trainer"):
                BtoA_D_train_op = training(BtoA_DLoss, BtoA_varD, scope='shared_variables/BtoA_Discriminator')
            with tf.name_scope("BtoA_Generator_trainer"):
                BtoA_G_train_op = training(BtoA_GLoss, BtoA_varG, scope='shared_variables/BtoA_generator')

            if use_identity_mapping:
                with tf.name_scope("LOSS"):
                    tf.summary.scalar("AtoBDLoss", AtoB_DLoss)
                    tf.summary.scalar("AtoBGLoss", AtoB_GLoss)
                    tf.summary.scalar("BtoADLoss", BtoA_DLoss)
                    tf.summary.scalar("BtoAGLoss", BtoA_GLoss)
                    tf.summary.scalar("{}Loss".format(cycle_consistency_loss), cycle_loss)
                    tf.summary.scalar("Identity_mapping_Loss", Identity_mapping_Loss)
            else:
                with tf.name_scope("visualizer"):
                    tf.summary.scalar("AtoBDLoss", AtoB_DLoss)
                    tf.summary.scalar("AtoBGLoss", AtoB_GLoss)
                    tf.summary.scalar("BtoADLoss", BtoA_DLoss)
                    tf.summary.scalar("BtoAGLoss", BtoA_GLoss)
                    tf.summary.scalar("{}Loss".format(cycle_consistency_loss), cycle_loss)

            summary_operation = tf.summary.merge_all()
            '''
            WHY? 아래 6줄의 코드를 적어 주지 않고, 학습을 하게되면, TEST부분에서 tf.train.import_meta_graph를 사용할 때 오류가 발생한다. 
            -> 단순히 그래프를 가져오고 가중치를 복원하는 것만으로는 안된다. 세션을 실행할때 인수로 사용할 변수에 대한 
            추가 접근을 제공하지 않기 때문에 아래와 같이 저장을 해놓은 뒤 TEST 시에 불러와서 다시 사용 해야한다. - 그렇지 않으면, JG.get_operations() 함수를
            사용해 출력된 모든 연산의 리스트에서 하나하나 찾아야 한다.
            필요한 변수가 있을 시 아래와 같이 추가해서 그래프를 새로 만들어 주면 된다.
            '''

            if norm_selection == "BN":
                for op in (A, B, AtoB_gene, BtoA_gene, BN_FLAG):
                    tf.add_to_collection("way", op)
            else:
                for op in (A, B, AtoB_gene, BtoA_gene):
                    tf.add_to_collection("way", op)

            # 아래와 같은 코드도 가능.
            # tf.add_to_collection('A', A)
            # tf.add_to_collection('B', B)
            # tf.add_to_collection('AtoB', AtoB_gene)
            # tf.add_to_collection('BtoA', BtoA_gene)

            # generator graph 구조를 파일에 쓴다.
            meta_save_file_path = os.path.join(model_name, 'Generator', 'Generator_Graph.meta')
            saver_generator.export_meta_graph(meta_save_file_path, collection_list=['way'])

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

                summary_writer = tf.summary.FileWriter(os.path.join('tensorboard', model_name), sess.graph)
                sess.run(A_iterator.initializer)
                sess.run(B_iterator.initializer)

                # A_length 와 B_length중 긴것을 택한다.
                data_length = A_length if A_length > B_length else B_length
                total_batch = int(data_length / batch_size)

                for epoch in tqdm(range(1, training_epochs + 1)):

                    # 논문에서 100 epoch가 넘으면 선형적으로 학습률(learning rate)을 감소시킨다고 했다. 1 epoch마다 0.99 씩 줄여보자
                    if epoch > weight_decay_epoch:
                        learning_rate *= learning_rate_decay

                    AtoB_LossD = 0
                    AtoB_LossG = 0
                    BtoA_LossD = 0
                    BtoA_LossG = 0

                    # 아래의 두 변수가 각각 0.5 씩의 값을 갖는게 가장 이상적이다.
                    AtoB_sigmoidD = 0
                    AtoB_sigmoidG = 0

                    # 아래의 두 변수가 각각 0.5 씩의 값을 갖는게 가장 이상적이다.
                    BtoA_sigmoidD = 0
                    BtoA_sigmoidG = 0

                    for i in range(total_batch):

                        # 입력 이미지가 256 x 256 이하이면, exit()
                        temp1, temp2 = sess.run([A, B])
                        if temp1.shape[1] < Inputsize_limit[0] or temp1.shape[2] < Inputsize_limit[1] or temp2.shape[
                            1] < Inputsize_limit[0] or temp2.shape[2] < Inputsize_limit[1]:
                            print("입력된 이미지 크기는 {}x{} 입니다.".format(temp.shape[1], temp.shape[2]))
                            print("입력되는 이미지 크기는 256x256 보다 크거나 같아야 합니다.")
                            print("강제 종료 합니다.")
                            exit(0)

                        if norm_selection == "BN":
                            # Generator Update
                            _, AtoB_G_Loss, AtoB_Dgene_simgoid = sess.run([AtoB_G_train_op, AtoB_GLoss, AtoB_Dgene],
                                                                          feed_dict={lr: learning_rate, BN_FLAG: True})
                            _, BtoA_G_Loss, BtoA_Dgene_simgoid = sess.run([BtoA_G_train_op, BtoA_GLoss, BtoA_Dgene],
                                                                          feed_dict={lr: learning_rate, BN_FLAG: True})
                        else:
                            # Generator Update
                            _, AtoB_G_Loss, AtoB_Dgene_simgoid = sess.run([AtoB_G_train_op, AtoB_GLoss, AtoB_Dgene],
                                                                          feed_dict={lr: learning_rate})
                            _, BtoA_G_Loss, BtoA_Dgene_simgoid = sess.run([BtoA_G_train_op, BtoA_GLoss, BtoA_Dgene],
                                                                          feed_dict={lr: learning_rate})

                        # image_pool 변수 사용할 때(단 batch_size=1 일 경우만), Discriminator Update
                        if image_pool and batch_size == 1:
                            fake_AtoB_gene, fake_BtoA_gene = imagepool(images=sess.run([AtoB_gene, BtoA_gene]))

                            # AtoB_gene, BtoA_gene 에 과거에 생성된 fake_AtoB_gene, fake_BtoA_gene를 넣어주자!!!
                            _, AtoB_D_Loss, AtoB_Dreal_simgoid = sess.run([AtoB_D_train_op, AtoB_DLoss, AtoB_Dreal],
                                                                          feed_dict={lr: learning_rate,
                                                                                     AtoB_gene: fake_AtoB_gene})
                            _, BtoA_D_Loss, BtoA_Dreal_simgoid = sess.run([BtoA_D_train_op, BtoA_DLoss, BtoA_Dreal],
                                                                          feed_dict={lr: learning_rate,
                                                                                     BtoA_gene: fake_BtoA_gene})
                        # image_pool 변수를 사용하지 않을 때, Discriminator Update
                        else:
                            if norm_selection == "BN":
                                _, AtoB_D_Loss, AtoB_Dreal_simgoid = sess.run([AtoB_D_train_op, AtoB_DLoss, AtoB_Dreal],
                                                                              feed_dict={lr: learning_rate,
                                                                                         BN_FLAG: True})
                                _, BtoA_D_Loss, BtoA_Dreal_simgoid = sess.run([BtoA_D_train_op, BtoA_DLoss, BtoA_Dreal],
                                                                              feed_dict={lr: learning_rate,
                                                                                         BN_FLAG: True})
                            else:
                                _, AtoB_D_Loss, AtoB_Dreal_simgoid = sess.run([AtoB_D_train_op, AtoB_DLoss, AtoB_Dreal],
                                                                              feed_dict={lr: learning_rate})
                                _, BtoA_D_Loss, BtoA_Dreal_simgoid = sess.run([BtoA_D_train_op, BtoA_DLoss, BtoA_Dreal],
                                                                              feed_dict={lr: learning_rate})

                        AtoB_LossD += (AtoB_D_Loss / total_batch)
                        AtoB_LossG += (AtoB_G_Loss / total_batch)
                        BtoA_LossD += (BtoA_D_Loss / total_batch)
                        BtoA_LossG += (BtoA_G_Loss / total_batch)

                        AtoB_sigmoidD += (np.mean(AtoB_Dreal_simgoid, axis=(1, 2, 3)) / total_batch)
                        AtoB_sigmoidG += (np.mean(AtoB_Dgene_simgoid, axis=(1, 2, 3)) / total_batch)
                        BtoA_sigmoidD += (np.mean(BtoA_Dreal_simgoid, axis=(1, 2, 3)) / total_batch)
                        BtoA_sigmoidG += (np.mean(BtoA_Dgene_simgoid, axis=(1, 2, 3)) / total_batch)

                        if norm_selection == "BN":
                            summary_str = sess.run(summary_operation, feed_dict={BN_FLAG: True})
                        else:
                            summary_str = sess.run(summary_operation)

                        summary_writer.add_summary(summary_str, global_step=epoch)

                        print("<<< {} epoch : {} batch running of {} total batch... >>>".format(epoch, i, total_batch))

                    print("<<< AtoB Discriminator mean output : {} / AtoB Generator mean output : {} >>>".format(
                        np.mean(AtoB_sigmoidD), np.mean(AtoB_sigmoidG)))
                    print("<<< BtoA Discriminator mean output : {} / BtoA Generator mean output : {} >>>".format(
                        np.mean(AtoB_sigmoidD), np.mean(AtoB_sigmoidG)))
                    print("<<< AtoB Discriminator Loss : {} / AtoB Generator Loss  : {} >>>".format(AtoB_LossD,
                                                                                                    AtoB_LossG))
                    print("<<< BtoA Discriminator Loss : {} / BtoA Generator Loss  : {} >>>".format(BtoA_LossD,
                                                                                                    BtoA_LossG))

                    if epoch % display_step == 0:

                        save_all_model_path = os.path.join(model_name, 'All')
                        save_generator_model_path = os.path.join(model_name, 'Generator')

                        if not os.path.exists(save_all_model_path):
                            os.makedirs(save_all_model_path)

                        if not os.path.exists(save_generator_model_path):
                            os.makedirs(save_generator_model_path)

                        saver_all.save(sess, save_all_model_path + "/", global_step=epoch,
                                       write_meta_graph=False)
                        saver_generator.save(sess, save_generator_model_path + "/",
                                             global_step=epoch,
                                             write_meta_graph=False)
                print("Optimization Finished!")

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
                A, B, AtoB_gene, BtoA_gene, BN_FLAG = tf.get_collection('way')
            else:
                A, B, AtoB_gene, BtoA_gene = tf.get_collection('way')

            # Test Dataset 가져오기
            dataset = Dataset(DB_name=DB_name, TFRecord=TFRecord,
                              use_TrainDataset=not TEST, inference_size=inference_size)
            A_iterator, A_next_batch, A_length, B_iterator, B_next_batch, B_length = dataset.iterator()
            A_tensor, B_tensor = A_next_batch, B_next_batch

            # A_length 와 B_length 중 짧은 것을 택한다.
            data_length = A_length if A_length < B_length else B_length

            with tf.Session(graph=JG) as sess:
                sess.run(A_iterator.initializer)
                sess.run(B_iterator.initializer)
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
                    # A_length 와 B_length 중 짧은 길이만큼만 생성
                    for i in range(data_length):
                        A_numpy, B_numpy = sess.run(
                            [A_tensor, B_tensor])  # 이런식으로 하는 것은 상당히 비효율적 -> tf.data.Dataset 에 더익숙해지고자!!!
                        # 입력 이미지가 256 x 256 이하이면, exit()
                        if A_numpy.shape[1] < Inputsize_limit[0] or A_numpy.shape[2] < Inputsize_limit[1] or \
                                B_numpy.shape[1] < Inputsize_limit[0] or B_numpy.shape[
                            2] < Inputsize_limit[1]:
                            print("<<< 입력된 이미지 크기는 {}x{} 입니다. >>>".format(A_numpy.shape[1], A_numpy.shape[2]))
                            print("<<< 입력되는 이미지 크기는 256x256 보다 크거나 같아야 합니다. >>>")
                            print("<<< 강제 종료 합니다. >>>")
                            exit(0)

                        if norm_selection == "BN":
                            AtoB_translated_image, BtoA_translated_image = sess.run([AtoB_gene, BtoA_gene],
                                                                                    feed_dict={A: A_numpy, B: B_numpy,
                                                                                               BN_FLAG: not not using_moving_variable})
                        else:
                            AtoB_translated_image, BtoA_translated_image = sess.run([AtoB_gene, BtoA_gene],
                                                                                    feed_dict={A: A_numpy, B: B_numpy})

                        # 순서 : 입력, 생성
                        visualize(model_name="AtoB" + model_name,
                                  named_images=[i, A_numpy[0], AtoB_translated_image[0]],
                                  save_path="AtoB" + save_path)
                        visualize(model_name="BtoA" + model_name,
                                  named_images=[i, B_numpy[0], BtoA_translated_image[0]],
                                  save_path="BtoA" + save_path)

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
                            print("<<< shape : ({}) >>>".format(shape))
                            f.write("<<< {}.npy >>>\n<<< shape : ({}) >>>\n\n".format(joined, shape))

                            # weight npy로 저장하기
                            np.save(os.path.join(numpy_weight_save_path, joined), reader.get_tensor(name))


if __name__ == "__main__":
    # 256x256 크기 이상의 다양한 크기의 이미지를 동시 학습 하는 것이 가능하다.(256 X 256으로 크기 제한을 뒀다.)
    # -> 단 batch_size =  1 일 때만 가능하다. - batch_size>=2 일때 여러사이즈의 이미지를 동시에 학습 하고 싶다면, 각각 따로 사이즈별로 Dataset을 생성 후 학습시키면 된다.
    # pix2pix GAN이나, Cycle gan이나 데이터셋 자체가 같은 크기의 이미지를 다루므로, 위 설명을 무시해도 된다.
    # TEST=False 시 입력 이미지의 크기가 256x256 미만이면 강제 종료한다.
    # TEST=True 시 입력 이미지의 크기가 256x256 미만이면 강제 종료한다.
    model(
        DB_name="horse2zebra",  # DB_name 은 "horse2zebra"에만 대비되어 있다.
        TEST=False,  # TEST=False -> Training or TEST=True -> TEST
        TFRecord=True,  # TFRecord=True -> TFRecord파일로 저장한후 사용하는 방식 사용 or TFRecord=False -> 파일에서 읽어오는 방식 사용
        Inputsize_limit=(256, 256),  # 입력되어야 하는 최소 사이즈를 내가 지정 - (256,256) 으로 하자
        filter_size=32,  # generator와 discriminator의 처음 layer의 filter 크기
        norm_selection="BN",  # IN - instance normalizaiton , BN -> batch normalization, NOTHING
        regularizer=" ",  # L1 or L2 정규화 -> 오버피팅 막기 위함
        scale=0.0001,  # L1 or L2 정규화 weight
        cycle_consistency_loss="L1",  # cycle loss -> L1 or L2
        cycle_consistency_loss_weight=10,  # cycle loss으 가중치
        optimizer_selection="Adam",  # optimizers_ selection = "Adam" or "RMSP" or "SGD"
        beta1=0.5, beta2=0.999,  # for Adam optimizer
        decay=0.999, momentum=0.9,  # for RMSProp optimizer
        use_identity_mapping=False,  # 논문에서는 painting -> photo DB 로 네트워크를 학습할 때 사용 - 우선은 False
        image_pool=True,  # discriminator 업데이트시 이전에 generator로 부터 생성된 이미지의 사용 여부
        image_pool_size=50,  # image_pool=True 라면 몇개를 사용 할지? 논문에선 50개 사용
        learning_rate=0.0002, training_epochs=1, batch_size=1, display_step=1,
        weight_decay_epoch=100,  # 몇 epoch 뒤에 learning_rate를 줄일지
        learning_rate_decay=0.99,  # learning_rate를 얼마나 줄일지
        inference_size=(512, 512),  # TEST=True 일때, inference 할 수 있는 최소의 크기를 256 x 256으로 크기 제한을 뒀다.
        using_moving_variable=False,  # TEST=True 일때, Moving Average를 Inference에 사용할지 말지 결정하는 변수
        only_draw_graph=False,  # TEST=False 일 때, 그래프만 그리고 종료할지 말지
        show_translated_image=True,  # TEST=True 일 때변환 된 이미지를 보여줄지 말지
        # 학습 완료 후 변환된 이미지가 저장될 폴더 2개가 생성 된다.(폴더 2개 이름 -> AtoB_translated_image , BtoA_translated_image )
        save_path="translated_image",  # TEST=True 일 때 변환된 이미지가 저장될 폴더
        weights_to_numpy=False)  # TEST=True 일 때 가중치를 npy 파일로 저장할지 말지
else:
    print("model imported")
