import glob
import os
import shutil

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm


def DataLoader(batch_size=None):
    # 1.데이터셋 읽기
    data = pd.read_excel("lotto.xlsx")
    data = np.asarray(data)
    input = data[1:, 1:]
    output = data[0:np.shape(data)[0] - 1, 1:]
    data = np.flipud(input).astype(np.float32)
    label = np.flip(output, axis=0).astype(np.float32)

    # Tensorflow 데이터셋 만들기 -
    '''tensorflow의 tf.data.Dataset utility를 사용했습니다. 
    직접 batch, shuffle등의 코드를 구현할 필요가 없습니다.!!! 
    '''
    dataset = tf.data.Dataset.from_tensor_slices((data.reshape((-1, 6)), label))  # 데이터셋 가져오기
    dataset = dataset.shuffle(len(data)).repeat().batch(batch_size)  # repeat() -> 계속 반복, batch() -> batchsize 지정
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next(), len(data) + 1


def model(TEST=False, optimizer_selection="Adam", learning_rate=0.0009, training_epochs=10000, batch_size=50,
          display_step=1,
          previous_first_prize_number=None, number_of_prediction=3, regularization='L2', scale=0.0001):
    model_name = "LottoNet"
    model_name = model_name + "reg" + regularization

    if TEST == False:
        if os.path.exists("tensorboard"):
            shutil.rmtree("tensorboard")

    def layer(input, weight_shape, bias_shape):
        weight_init = tf.contrib.layers.xavier_initializer(uniform=False)
        bias_init = tf.zeros_initializer()
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

    def inference(x):
        with tf.variable_scope("fully1"):
            fully_1 = tf.nn.relu(layer(tf.reshape(x, (-1, 6)), [6, 100], [100]))
        with tf.variable_scope("fully2"):
            fully_2 = tf.nn.relu(layer(fully_1, [100, 100], [100]))
        with tf.variable_scope("fully3"):
            fully_3 = tf.nn.relu(layer(fully_2, [100, 100], [100]))
        with tf.variable_scope("fully4"):
            fully_4 = tf.nn.relu(layer(fully_3, [100, 100], [100]))
        with tf.variable_scope("fully5"):
            fully_5 = tf.nn.relu(layer(fully_4, [100, 100], [100]))
        with tf.variable_scope("fully6"):
            fully_6 = tf.nn.relu(layer(fully_5, [100, 100], [100]))
        with tf.variable_scope("prediction"):
            return layer(fully_6, [100, 6], [6])

    def loss(output, y):
        l2 = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(output, tf.reshape(y, (-1, 6)))), axis=1))
        train_loss = tf.reduce_mean(l2)
        return train_loss

    def training(cost, global_step):
        tf.summary.scalar("train_cost", cost)
        if optimizer_selection == "Adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif optimizer_selection == "RMSP":
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        elif optimizer_selection == "SGD":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_operation = optimizer.minimize(cost, global_step=global_step)
        return train_operation

    if not TEST:
        # print(tf.get_default_graph()) #기본그래프이다.
        JG = tf.Graph()  # 내 그래프로 설정한다.- 혹시라도 나중에 여러 그래프를 사용할 경우를 대비
        with JG.as_default():  # as_default()는 JG_Graph를 기본그래프로 설정한다.

            next_batch, data_length = DataLoader(batch_size)
            # x, y 도 tensor 이다.
            # tf.data.Dataset을 사용함으로써 graph의 일부분이 되었다. -> feed_dict 으로 값을 넣어줄 필요가 없다.
            x, y = next_batch

            with tf.variable_scope("shared_variables", reuse=tf.AUTO_REUSE) as scope:
                with tf.name_scope("inference"):
                    output = inference(x)
                # or scope.reuse_variables()

            # Adam optimizer의 매개변수들을 저장하고 싶지 않다면 여기에 선언해야한다.
            with tf.name_scope("saver"):
                saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=3)
            with tf.name_scope("loss"):
                global_step = tf.Variable(0, name="global_step", trainable=False)
                cost = loss(output, y)
            with tf.name_scope("trainer"):
                train_operation = training(cost, global_step)
            with tf.name_scope("tensorboard"):
                summary_operation = tf.summary.merge_all()

            '''
            WHY? 아래 3줄의 코드를 적어 주지 않고, 학습을 하게되면, TEST부분에서 tf.train.import_meta_graph를 사용할 때 오류가 발생한다. 
            -> 단순히 그래프를 가져오고 가중치를 복원하는 것만으로는 안된다. 세션을 실행할때 인수로 사용할 변수에 대한 
            추가 접근을 제공하지 않기 때문에 아래와 같이 저장을 해놓은 뒤 TEST 시에 불러와서 다시 사용 해야한다.
            '''
            tf.add_to_collection('x', x)
            tf.add_to_collection('output', output)
            # graph 구조를 파일에 쓴다.
            saver.export_meta_graph(os.path.join(model_name, "Lotto_Graph.meta"), collection_list=['x', 'output'])

        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(graph=JG, config=config) as sess:
            print("initializing!!!")
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(model_name)

            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                print("Restore {} checkpoint!!!".format(os.path.basename(ckpt.model_checkpoint_path)))
                saver.restore(sess, ckpt.model_checkpoint_path)

            summary_writer = tf.summary.FileWriter(os.path.join("tensorboard"), sess.graph)
            for epoch in tqdm(range(training_epochs)):
                avg_cost = 0.
                total_batch = int(data_length / batch_size)
                for i in range(total_batch):
                    _, minibatch_cost = sess.run([train_operation, cost])
                    avg_cost += (minibatch_cost / total_batch)

                print("L2 cost : {}".format(avg_cost))
                if epoch % display_step == 0:
                    summary_str = sess.run(summary_operation)
                    summary_writer.add_summary(summary_str, global_step=sess.run(global_step))
                    if not os.path.exists(model_name):
                        os.makedirs(model_name)
                    saver.save(sess, model_name + "/", global_step=sess.run(global_step),
                               write_meta_graph=False)
            print("Optimization Finished!")

    else:
        tf.reset_default_graph()
        meta_path = glob.glob(os.path.join(model_name, '*.meta'))
        if len(meta_path) == 0:
            print("<<< Lotto Graph가 존재 하지 않습니다. >>>")
            exit(0)
        else:
            print("<<< Lotto Graph가 존재 합니다. >>>")

        # print(tf.get_default_graph()) #기본그래프이다.
        JG = tf.Graph()  # 내 그래프로 설정한다.- 혹시라도 나중에 여러 그래프를 사용할 경우를 대비
        with JG.as_default():  # as_default()는 JG를 기본그래프로 설정한다.

            saver = tf.train.import_meta_graph(meta_path[0], clear_devices=True)  # meta graph 읽어오기
            if saver == None:
                print("<<< meta 파일을 읽을 수 없습니다. >>>")
                exit(0)

            '''
            WHY? 아래 2줄의 코드를 적어 주지 않으면 오류가 발생한다. 
            -> 단순히 그래프를 가져오고 가중치를 복원하는 것만으로는 안된다. 세션을 실행할때 인수로 사용할 변수에 대한 
            추가 접근을 제공하지 않기 때문에 아래와 같이 get_colltection으로 입,출력 변수들을 불러와서 다시 사용 해야 한다.
            '''
            x = tf.get_collection('x')[0]
            output = tf.get_collection('output')[0]

            with tf.Session(graph=JG) as sess:
                sess.run(tf.global_variables_initializer())
                ckpt = tf.train.get_checkpoint_state(model_name)
                if ckpt == None:
                    print("<<< checkpoint file does not exist>>>")
                    print("<<< Exit the program >>>")
                    exit(0)
                if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                    print("Restore {} checkpoint!!!".format(os.path.basename(ckpt.model_checkpoint_path)))
                    saver.restore(sess, ckpt.model_checkpoint_path)

                prediction_number = sess.run(output,
                                             feed_dict={x: np.asarray(previous_first_prize_number).reshape(-1, 6)})

                prediction_number_rint = np.clip(np.rint(prediction_number[-1]), a_min=0, a_max=45)
                prediction_number_floor = np.clip(np.floor(prediction_number[-1]), a_min=0, a_max=45)
                prediction_number_ceil = np.clip(np.ceil(prediction_number[-1]), a_min=0, a_max=45)

                except_result = np.asarray(list(set(list(prediction_number_floor)
                                                    + list(prediction_number_rint)
                                                    + list(prediction_number_ceil)
                                                    + list(previous_first_prize_number[-1]))),
                                           dtype=np.int32)

                except_result = np.sort(except_result)
                if 0 in except_result:
                    except_result = np.delete(except_result, 0)
                select_number = np.delete(np.arange(1, 46), except_result - 1)
                np.random.shuffle(select_number)

                with open("prediction.txt", "w") as f:
                    for i in range(number_of_prediction):
                        result = np.sort(np.random.choice(select_number, 6, replace=False))  # replace=False -> 중복 허용 x
                        print("당첨 예측 번호-{} : {}".format(i + 1, result))
                        f.write("당첨 예측 번호-{} : {}\n".format(i + 1, result))


if __name__ == "__main__":
    # optimizers_ selection = "Adam" or "RMSP" or "SGD"
    model(TEST=False, optimizer_selection="Adam", learning_rate=0.0009, training_epochs=100000, batch_size=256,
          display_step=100,
          # 전 회차 당첨번호 6자리 입력
          # 반드시 이차원 배열로 선언
          previous_first_prize_number=[[2, 21, 28, 38, 42, 45]], number_of_prediction=5, regularization='L2',
          scale=0.0001)
else:
    print("model imported")
