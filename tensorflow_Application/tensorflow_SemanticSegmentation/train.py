import shutil
import timeit

from core.dataset import *
from core.function_list import *
from core.model.network import *


class Model(object):

    def __init__(self,
                 DB_choice_percentage=0.1,
                 network="UNET",
                 variable_scope="Segmentation",
                 Inputsize_limit=(256, 256),
                 init_filter_size=32,
                 regularizer="L1",
                 scale=0.0001,
                 Dropout_rate=0.5,
                 loss_selection="pixelwise_softmax_cross_entropy",
                 loss_weight=1,
                 optimizer_selection="Adam",
                 beta1=0.5, beta2=0.999,
                 decay=0.999, momentum=0.9,
                 input_range="0~1",
                 Augmentation_algorithm=3,
                 learning_rate=0.0002,
                 lr_decay_epoch=100,
                 lr_decay=0.99,
                 learning_time=100,
                 training_epochs=200,
                 batch_size=1,
                 save_step=1,
                 using_latest_weight=True,
                 WeightSelection=200,
                 Accesskey="way",
                 only_draw_graph=False,
                 class_number=None,
                 *args, **kwargs):

        self.DB_choice_percentage = DB_choice_percentage
        self.network = network
        self.variable_scope = variable_scope
        self.Inputsize_limit = Inputsize_limit
        self.input_range = input_range
        self.init_filter_size = init_filter_size
        self.regularizer = regularizer
        self.scale = scale
        self.Dropout_rate = Dropout_rate
        self.loss_selection = loss_selection
        self.loss_weight = loss_weight
        self.optimizer_selection = optimizer_selection
        self.beta1 = beta1
        self.beta2 = beta2
        self.decay = decay
        self.momentum = momentum

        self.Augmentation_algorithm = Augmentation_algorithm
        self.learning_rate = learning_rate
        self.lr_decay_epoch = lr_decay_epoch
        self.lr_decay = lr_decay
        self.learning_time = learning_time
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.save_step = save_step
        self.using_latest_weight = using_latest_weight
        self.WeightSelection = WeightSelection
        self.Accesskey = Accesskey
        self.only_draw_graph = only_draw_graph
        self.class_number = class_number
        self.model_name = str(init_filter_size)

        self.model_name += self.network
        if self.loss_selection == "pixelwise_softmax_cross_entropy":
            print("<<< Autoencoder with softmax cross entropy loss >>>")
            self.model_name += "SCEL"
        elif self.loss_selection == "soft_dice":
            print("<<< Autoencoder with soft dice loss >>>")
            self.model_name += "SD"
        elif self.loss_selection == "squared_soft_dice":
            print("<<< Autoencoder with soft dice loss >>>")
            self.model_name += "SSD"
        else:
            raise ValueError("Unknown loss")
            # print("<<< 선택된 Loss가 없습니다. softmax_cross_entropy와 soft_dice 중에 선택 해주세요 >>>")
            # print("<<< 강제 종료합니다. >>>")
            # exit(0)

        self.model_name = self.model_name + "IN"
        if regularizer == "L1" or regularizer == "L2":
            self.model_name = self.model_name + "reg" + regularizer

        # GRPC를 위함
        self.model_name = self.model_name + self.input_range

        if not os.path.exists(self.model_name):
            os.makedirs(self.model_name)

        if os.path.exists("tensorboard/{}".format(self.model_name)):
            shutil.rmtree("tensorboard/{}".format(self.model_name))

        # class_number > 7 이상인 경우는 검은색으로 기본 설정하자.(이게 싫으면 색을 직접 추가하시길)
        # segmentation용 color - 우선 최대 9개 색깔 지정
        # RGB 순서
        self.color = defaultdict(lambda: [0, 0, 0])
        self.color[0] = [250, 237, 125]  # 노
        self.color[1] = [255, 0, 0]  # 빨
        self.color[2] = [255, 255, 255]  # 흰색
        self.color[3] = [255, 130, 36]  # 주
        self.color[4] = [0, 0, 0]  # 검
        self.color[5] = [0, 255, 0]  # 초
        self.color[6] = [0, 0, 255]  # 파
        self.color[7] = [3, 0, 102]  # 남
        self.color[8] = [128, 65, 217]  # 보

        # 그래프 그리기
        self.make_graph()

    def select_optimizer(self, cost, var_list, scope=None):

        if self.regularizer == "L1" or self.regularizer == "L2":
            self.cost = tf.add_n([cost] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=scope))

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)):
            if self.optimizer_selection == "Adam":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.beta1, beta2=self.beta2)
            elif self.optimizer_selection == "RMSP":
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=self.decay,
                                                           momentum=self.momentum)
            elif self.optimizer_selection == "SGD":
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
            elif self.optimizer_selection == "Momentum":
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=self.momentum,
                                                            use_nesterov=False)
            elif self.optimizer_selection == "Nesterov":
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=self.momentum,
                                                            use_nesterov=True)
            train_operation = self.optimizer.minimize(cost, var_list=var_list)

        return train_operation

    def make_graph(self):

        # print(tf.get_default_graph()) #기본그래프이다.
        self.graph = tf.Graph()  # 내 그래프로 설정한다.- 혹시라도 나중에 여러 그래프를 사용할 경우를 대비
        with self.graph.as_default():  # as_default()는 graph를 기본그래프로 설정한다.

            with tf.name_scope("Dataset"):

                # `GraphKeys.TRAINABLE_VARIABLES` 에 추가될 필요가 없으니 trainable=False 로 한다.
                with tf.name_scope("Learning_rate"):
                    self.lr = tf.Variable(initial_value=self.learning_rate, trainable=False, dtype=tf.float32)

                if self.only_draw_graph == False:
                    dataset = Dataset(batch_size=self.batch_size, use_TrainDataset=True,
                                      input_range=self.input_range,
                                      Augmentation_algorithm=self.Augmentation_algorithm)
                    self.next_batch, self.data_length = dataset.iterator()

                    x = self.next_batch[0]
                    target = self.next_batch[1]

                    information = np.array([self.class_number, self.input_range, self.Accesskey])
                    np.save(os.path.join(self.model_name, "information.npy"), information)

                else:
                    print("self.only_draw_graph == True 일 경우 그래프만 그립니다.")
                    print("이와 같이 그래프만 그릴 경우에는 class_number를 수동으로 선택해줘야 합니다.")
                    print("현재는 2개의 클래스만 Segmentation 하므로 '2' 로 설정 되어 있습니다.")
                    print("나중에 N개의 클래스로 확장하고, 그래프를 다시 그려야 할 경우, class_number를 N으로 바꾸셔야 합니다.")
                    x = tf.placeholder(dtype=tf.float32, shape=(None))
                    target = tf.placeholder(dtype=tf.float32, shape=(None))

            with tf.name_scope("Dropout_rate"):
                self.keep_probability = tf.placeholder(tf.float32, shape=None)

            # 알고리즘
            net = Network(class_number=self.class_number, init_filter_size=self.init_filter_size,
                          regularizer=self.regularizer,
                          scale=self.scale,
                          keep_probability=self.keep_probability)

            with tf.variable_scope(self.variable_scope, reuse=tf.AUTO_REUSE) as scope:
                with tf.name_scope("Segmentation"):
                    if self.network == "UNET":
                        G, conv_list = net.UNET4x4(images=x, name=self.network)
                    else:
                        raise ValueError("<<< Segmentation Network를 선택해 주세요. >>>")
                        # print("<<< Segmentation Network를 선택해 주세요. >>>")
                        # exit(0)

            var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                      scope='{}/{}'.format(self.variable_scope, self.network))

            # optimizer의 매개변수들을 저장하고 싶지 않다면 여기에 선언해야한다.
            with tf.name_scope("Saver"):
                self.saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=self.training_epochs)

            # 나중에 N개 클래스로 확장시 클래스 개수/전체데이터 개수를 고려한 weight를 구해서 곱해주시길.
            # 더 잘된 다고 함.

            # 1. pixel wise softmax cross entropy
            if self.loss_selection == "pixelwise_softmax_cross_entropy":
                with tf.name_scope("{}_loss".format(self.loss_selection)):
                    # loss 구현하기
                    self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=target, logits=G, dim=-1)
                    self.loss = tf.reduce_mean(tf.multiply(self.loss, self.loss_weight))

            # 2. soft dice
            elif self.loss_selection == "soft_dice":
                '''
                https://www.researchgate.net/post/In_segmentation_task_pixel-wise_softmax_or_dice
                Generally, both methods can be applied to train a pixel-wise segmentation model. 
                The suitable usage by considering natures of two different cost function is that using dice-coef 
                if you cannot guarantee the balanced distribution of all classes of training data, and vice versa.
                
                # https://www.jeremyjordan.me/semantic-segmentation/
                In order to quantify |A| and |B|, some researchers use the simple sum whereas other researchers prefer 
                to use the squared sum for this calculation.  I don't have the practical experience to know 
                which performs better empirically over a wide range of tasks, 
                so I'll leave you to try them both and see which works better.
                
                In case you were wondering, there's a 2 in the numerator in calculating the Dice coefficient 
                because our denominator "double counts" the common elements between the two sets.
                In order to formulate a loss function which can be minimized, 
                we'll simply use 1−Dice. This loss function is known as the soft Dice loss 
                because we directly use the predicted probabilities instead of thresholding and converting them into a binary mask.
                '''
                # unbalanced한 DB에는 soft_dice를 사용하기
                with tf.name_scope("{}_loss".format(self.loss_selection)):
                    prediction = tf.nn.softmax(logits=G, axis=-1)
                    # batch, class 축만 남기고 계산하기 - 분자
                    intersection = tf.reduce_sum(prediction * target, axis=(1, 2))
                    # |A| , |B| 이고
                    # batch, class 축만 남기고 계산하기 - 분모
                    union = tf.reduce_sum(prediction, axis=(1, 2)) + tf.reduce_sum(target, axis=(1, 2))

                    # 이건 |A|^2 , |B|^2 이다
                    # union = eps + tf.reduce_sum(tf.suare(prediction), axis=(1, 2) + tf.reduce_sum(tf.square(target), axis=(1, 2)
                    self.loss = 1 - tf.reduce_mean(tf.divide(2 * intersection, union + 1e-9))

            # 3. squared soft dice
            elif self.loss_selection == "squared_soft_dice":

                with tf.name_scope("{}_loss".format(self.loss_selection)):
                    prediction = tf.nn.softmax(logits=G, axis=-1)
                    # batch, class 축만 남기고 계산하기 - 분자
                    intersection = tf.reduce_sum(prediction * target, axis=(1, 2))
                    # 이건 |A| , |B| 이고

                    # batch, class 축만 남기고 계산하기 - 분모
                    # |A|^2 , |B|^2 이다
                    union = tf.reduce_sum(tf.square(prediction), axis=(1, 2)) + tf.reduce_sum(tf.square(target),
                                                                                              axis=(1, 2))
                    self.loss = 1 - tf.reduce_mean(tf.divide(2 * intersection, union + 1e-9))

            with tf.name_scope("segmentation_trainer"):
                self.G_train_op = self.select_optimizer(self.loss, var,
                                                        scope='{}/{}'.format(self.variable_scope, self.network))

            with tf.name_scope("Loss"):
                tf.summary.scalar("{}Loss".format(self.loss_selection), self.loss)

            with tf.name_scope("evaluation"):

                G = tf.nn.softmax(G, axis=-1)  # Depth 축으로 Softmax 하기
                conv_list.append(G)

                # 기본이 int64 라서 int32 로 설정해주자.
                t = tf.argmax(target, -1, output_type=tf.int32)
                p = tf.argmax(G, -1, output_type=tf.int32)

                # mean Intersection Over Union 계산
                self.mIOU, self.update_op_mIOU = tf.metrics.mean_iou(labels=t, predictions=p,
                                                                     num_classes=self.class_number)
                tf.summary.scalar("mIOU", self.mIOU)

                # pixel accuracy 계산
                true = tf.ones_like(t)
                false = tf.zeros_like(t)
                temp = tf.where(tf.equal(t, p), x=true, y=false)

                numerator = tf.count_nonzero(temp, dtype=tf.int32)
                denominator = tf.size(temp, out_type=tf.int32)

                self.pacc = tf.divide(numerator, denominator)
                tf.summary.scalar("pixel accuracy", self.pacc)

            with tf.name_scope("class_viewer"):
                # batch = N 인 경우, 첫번째 DB의 입력만 텐서보드에 출력된다..
                tf.summary.image("Input", x[0:1, :, :, :])
                for i in range(self.class_number):
                    # batch = N 인 경우, 첫번째 DB의 출력만 class 별로 텐서보드에 출력된다.
                    # N개의 DB에 대해서 class별로 다 나오게 할 수도 있지만, 너무 많다.
                    thresholded = G[0:1, :, :, i:i + 1]  # 0 or 1로 만든다.
                    tf.summary.image("class{}".format(i), thresholded)

            with tf.name_scope("all_viewer"):
                # batch = N 인 경우, 첫번째 DB의 segmented_image만 텐서보드에 출력된다.
                # Tensorboard 에 세그멘테이션 이미지 그리기

                input = x[0:1, :, :, :]

                t_expanded = tf.expand_dims(t, axis=-1)[0:1, :, :, :]  # or target[:, :, :, tf.newaxis]
                p_expanded = tf.expand_dims(p, axis=-1)[0:1, :, :, :]  # or predition[:, :, :, tf.newaxis]
                t_tiled = tf.tile(t_expanded, tf.constant([1, 1, 1, 3]))
                p_tiled = tf.tile(p_expanded, tf.constant([1, 1, 1, 3]))
                
                # 색칠될 이미지 공간
                t_temp = tf.zeros_like(t_tiled, dtype=tf.uint8)
                p_temp = tf.zeros_like(p_tiled, dtype=tf.uint8)

                for i in range(self.class_number):
                    # tf.where 의 x,y에 들어갈 인자(np.where에서는 broadcast가 되는데, tensorflow에서는 안되기 때문에
                    # 아래의 코드를 구현함.

                    color = tf.convert_to_tensor(np.reshape(self.color[i], (1, 1, 1, -1)), dtype=tf.uint8)
                    true = tf.tile(color, tf.shape(t_expanded))
                    false = tf.zeros_like(true, dtype=tf.uint8)

                    # condition, x, y 가 같은 모양이어야 한다.
                    t_temp += tf.where(tf.equal(t_tiled, i), x=true, y=false)
                    p_temp += tf.where(tf.equal(p_tiled, i), x=true, y=false)

                # 전부다 0~1범위의 float32로 바꾸자 - float32, int32, int32 를 붙이는 것은 안된다.
                t_temp = tf.divide(tf.cast(t_temp, dtype=tf.float32), 255)
                p_temp = tf.divide(tf.cast(p_temp, dtype=tf.float32), 255)
                concatenated = tf.concat((input, t_temp, p_temp), axis=2)
                tf.summary.image("result", concatenated)

            self.summary_operation = tf.summary.merge_all()

            '''
            WHY? 아래 2줄의 코드를 적어 주지 않고, 학습을 하게되면, TEST부분에서 tf.train.import_meta_graph를 사용할 때 오류가 발생한다. 
            -> 단순히 그래프를 가져오고 가중치를 복원하는 것만으로는 안된다. 세션을 실행할때 인수로 사용할 변수에 대한 
            추가 접근을 제공하지 않기 때문에 아래와 같이 저장을 해놓은 뒤 TEST 시에 불러와서 다시 사용 해야한다. - 그렇지 않으면, JG.get_operations() 함수를
            사용해 출력된 모든 연산의 리스트에서 하나하나 찾아야한다. 필요한 변수가 있을 시 아래와 같이 추가해서 그래프를 새로 만들어 주면 된다.
            '''

            for op in [x, target, G, self.loss, self.keep_probability] + conv_list:
                tf.add_to_collection(self.Accesskey, op)

            # segmentation graph 구조를 파일에 쓴다.
            meta_save_file_path = os.path.join(self.model_name, 'segmentation_Graph.meta')
            self.saver.export_meta_graph(meta_save_file_path, collection_list=[self.Accesskey])

            if self.only_draw_graph:
                print('<<< Generator_Graph.meta 파일만 저장하고 종료합니다. >>>')
                exit(0)

            self.config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
            # config.gpu_options.allow_growth = True
            # config.gpu_options.per_process_gpu_memory_fraction = 0.1

    def train(self):

        with tf.Session(graph=self.graph, config=self.config) as sess:
            print("\n<<< weight initializing!!! >>>")
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())  # mion , pixel accuracy를 위해서 초기화 해줘야 한다.
            ckpt = tf.train.get_checkpoint_state(self.model_name)
            if (ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path)):
                print("<<< all variable retored except for optimizer parameter >>>")
                if self.using_latest_weight:
                    self.saver.restore(sess, ckpt.model_checkpoint_path)
                    Latest_WeightSelection = os.path.basename(ckpt.model_checkpoint_path).split("-")[-1]
                    print("<<< Restore {} checkpoint!!! >>>".format(Latest_WeightSelection))
                    start = int(Latest_WeightSelection)
                else:
                    weight_list = ckpt.all_model_checkpoint_paths
                    Condition = False
                    for i, wl in enumerate(weight_list):
                        if int(os.path.basename(wl).split("-")[-1]) == self.WeightSelection:
                            self.saver.restore(sess, wl)
                            print("<<< Restore {} checkpoint!!! >>>".format(wl))
                            Condition = True
                            break
                        else:
                            Condition = False
                    start = self.WeightSelection

                    if Condition == False:
                        print("<<< 선택한 {} 가중치가 없습니다. >>>".format(self.WeightSelection))
                        print("<<< 강제 종료합니다. >>>")
                        exit(0)
            else:
                start = 0

            summary_writer = tf.summary.FileWriter(os.path.join("tensorboard", self.model_name), sess.graph)

            # percentage별 학습
            for epoch in range(start + 1, self.training_epochs + 1, 1):

                if epoch > self.lr_decay_epoch:
                    self.learning_rate *= self.lr_decay

                count = 0
                Loss = 0
                total_batch = int(self.data_length)
                total_time = 0
                for i in range(1, total_batch + 1, 1):

                    # 입력 이미지가 Inputsize_limit[0] x Inputsize_limit[1] 이하이면, exit()
                    # 여기서 dataloader 자체가 멀티스레드로 동작하기 때문에, 내가 생각한 순서로 데이터가 들어오지 않는다.
                    # 그냥 크기 제한을 하는 용도로만 생각하면 좋겠다.
                    # temp = sess.run(x)
                    # if temp.shape[1] < Inputsize_limit[0] or temp.shape[2] < Inputsize_limit[1]:
                    #     print("<<< 입력된 이미지 크기는 {} x {} 입니다. >>>".format(temp.shape[1], temp.shape[2]))
                    #     print("<<< 입력되는 이미지 크기는 {} x {} 보다 크거나 같아야 합니다. >>>".format(Inputsize_limit[0],
                    #                                                                 Inputsize_limit[1]))
                    #     print("<<< 강제 종료 합니다. >>>")
                    #     exit(0)

                    DB_choice_percentage = np.clip(self.DB_choice_percentage, 0, 1)
                    if np.random.uniform(low=0.0, high=1.0) <= DB_choice_percentage:

                        count += 1
                        start_time = timeit.default_timer()

                        '''
                        mean IOU 사용법 -> self.update_op_mIOU(confusion matrix)b를
                        sess.run 한 후, self.mIOU 를 sess.run 한다. 
                        For estimation of the metric over a stream of data, the function creates an
                        `update_op` operation that updates these variables and returns the `mean_iou`.
                        '''

                        _, loss, _, pixelacc = sess.run([self.G_train_op, self.loss, self.update_op_mIOU, self.pacc],
                                                        feed_dict={self.keep_probability: self.Dropout_rate})

                        meanIOU = sess.run(self.mIOU)

                        end_time = timeit.default_timer()
                        total_time += (end_time - start_time)

                        # loss
                        Loss += (loss / (total_batch * self.DB_choice_percentage))

                        if count % self.learning_time == 0:
                            print("<<< {} epoch : {} batch running of {} total batch... >>>".format(epoch, count,
                                                                                                    total_batch * self.DB_choice_percentage))
                            print("<<< DB {}개 학습에 걸리는 시간 : {:0.3f}ms >>>".format(self.learning_time, total_time * 1000))
                            total_time = 0

                print("<<< {} / {} epoch >>>".format(epoch, self.training_epochs))
                print("<<< {} loss : {} >>>".format(self.loss_selection, Loss))
                print("<<< pixel accuracy : {:0.3f}%".format(pixelacc * 100))
                print("<<< meanIOU : {:0.3f}%".format(meanIOU * 100))

                if epoch % self.save_step == 0:
                    summary_str = sess.run(self.summary_operation, feed_dict={self.keep_probability: self.Dropout_rate})
                    summary_writer.add_summary(summary_str, global_step=epoch)

                    if not os.path.exists(self.model_name):
                        os.makedirs(self.model_name)

                    self.saver.save(sess, self.model_name + "/", global_step=epoch,
                                    write_meta_graph=False)

            print("<<< Optimization Finished! >>>")
