import glob
import os
import shutil
import time

import cv2
import gym
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm


class ReplacyMemory(object):

    def __init__(self, maxlen, batch_size, with_replacement):

        self.maxlen = maxlen
        self.buffer = np.empty(shape=maxlen, dtype=np.object)  # dtype = np.object인게 중요하다.

        self.index = 0
        self.length = 0
        self.batch_size = batch_size
        self.with_replacement = with_replacement
        self.sequence_state = []

    def remember(self, sequence_state, action, reward, next_state, gamestate):

        self.sequence_state = np.concatenate((sequence_state[:, :, 1:], next_state[:, :, np.newaxis]), axis=-1)
        self.buffer[self.index] = [sequence_state, action, reward, self.sequence_state, gamestate]
        self.length = min(self.length + 1, self.maxlen)
        self.index = (self.index + 1) % self.maxlen

    @property
    def next_sequence_state(self):
        return self.sequence_state

    @property
    def sample(self):
        if self.with_replacement:
            indices = np.random.randint(self.length, size=self.batch_size)
        else:
            indices = np.random.permutation(self.length)[:self.batch_size]
        return self.buffer[indices]


class model(object):

    def __init__(self,
                 model_name="BreakoutDeterministic-v4",
                 training_display=(True, 100000),
                 training_step=200000000,
                 training_start_point=10000,
                 training_interval=4,
                 rememorystackNum=500000,
                 save_step=10000,
                 copy_step=10000,
                 framesize=4,
                 learning_rate=0.00025,
                 momentum=0.95,
                 egreedy_max=1,
                 egreedy_min=0.1,
                 egreedy_step=1000000,
                 discount_factor=0.99,
                 batch_size=32,
                 with_replacement=True,
                 only_draw_graph=False,
                 SaveGameMovie=True):

        # 환경 만들기
        self.model_name = model_name + "_IC" + str(framesize)  # IC -> Input Channel

        self.env = gym.make(model_name)  # train , test
        self.val_env = gym.make(model_name)  # val

        self.display = training_display[0]
        self.display_step = training_display[1]

        self.SaveGameMovie = SaveGameMovie

        # 학습 하이퍼파라미터
        self.framesize = framesize
        self.training_step = training_step
        self.training_start_point = training_start_point
        self.training_interval = training_interval
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.egreedy_min = egreedy_min
        self.egreedy_max = egreedy_max
        self.egreedy_step = egreedy_step

        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.save_step = save_step
        self.copy_step = copy_step
        self.only_draw_graph = only_draw_graph

        # 재현 메모리
        self.stacked_state = []  # 연속 관측을 담아두기 위한 변수
        self.with_replacement = with_replacement
        self.rememorystackNum = rememorystackNum

        self.RM = ReplacyMemory(maxlen=self.rememorystackNum, batch_size=self.batch_size,
                                with_replacement=self.with_replacement)

        # DQN 연산그래프 그리기
        self._build_graph()

    def __repr__(self):
        print("{} With DDQN".format(self.model_name))

    @property
    def _action_space_number(self):
        return self.env.action_space.n

    @property
    def _sample_memories(self):
        # 상태, 행동, 보상, 다음 상태, 게임지속여부
        cols = [[], [], [], [], []]
        for memory in self.RM.sample:
            for col, value in zip(cols, memory):
                col.append(value)

        cols = [np.array(col) for col in cols]

        state = cols[0]
        action = cols[1]
        reward = cols[2].reshape(-1, 1)  # 형태 맞춰주기
        next_state = cols[3]
        gamestate = cols[4].reshape(-1, 1)  # 형태 맞춰주기

        return state, action, reward, next_state, gamestate

    def _epsilon_greedy(self, Qvalue, step):

        # off policy 요소
        # 훈련 스텝 전체에 걸쳐서 epsilon을 1.0 에서 0.1로 감소 시킨다.
        epsilon = self.egreedy_max - (self.egreedy_max - self.egreedy_min) * (step / self.egreedy_step)
        if np.random.rand() < epsilon:  # epsilon 확률로 랜덤하게 행동
            return np.random.randint(self._action_space_number)
        else:
            return np.argmax(Qvalue)  # 1 - epsilon 확률로 랜덤하게 행동

    # DDQN의 연산량을 줄이고 훈련속도를 향상시키기
    def _data_preprocessing(self, obs):

        # 84 x 84 gray로 만들기
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        obs = cv2.resize(obs, dsize=(84, 84))
        return obs.astype(np.uint8)

    def _concatenated_state(self, state):
        listed_state = [self._data_preprocessing(state)[:, :, np.newaxis] for _ in range(self.framesize)]
        concatenated_state = np.concatenate(listed_state, axis=-1)
        return concatenated_state

    def _DQN(self, inputs, name):

        # N X 84 x 84 x 4
        initializer = tf.contrib.layers.variance_scaling_initializer()
        with tf.variable_scope(name) as scope:
            conv1 = tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=(8, 8), strides=(4, 4), padding='valid',
                                     activation=tf.nn.relu, use_bias=True,
                                     kernel_initializer=initializer)  # N X 20 X 20 X 32
            conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=(4, 4), strides=(2, 2), padding='valid',
                                     activation=tf.nn.relu, use_bias=True,
                                     kernel_initializer=initializer)  # N X 9 X 9 X 64
            conv3 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid',
                                     activation=tf.nn.relu, use_bias=True,
                                     kernel_initializer=initializer)  # N X 7 X 7 X 64
            hidden = tf.layers.dense(tf.reshape(conv3, shape=(-1, 7 * 7 * 64)), 512, activation=tf.nn.relu,
                                     use_bias=True, kernel_initializer=initializer)
            output = tf.layers.dense(hidden, self._action_space_number, activation=None, use_bias=True,
                                     kernel_initializer=initializer)

            # train_vars = tf.trainable_variables(scope = scope.name)
            train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
            train_vars_dictionary = {var.name[len(scope.name):]: var for var in train_vars}
            return output, train_vars_dictionary

    def _build_graph(self):

        self.Graph = tf.Graph()
        with self.Graph.as_default():

            # model input
            self.state = tf.placeholder(tf.float32, shape=[None, None, None, self.framesize])
            self.action = tf.placeholder(tf.int32, shape=None)

            self.target = tf.placeholder(tf.float32, shape=None)

            # tensorboard
            self.rewards = tf.placeholder(tf.float32, shape=None)
            self.Qvalues = tf.placeholder(tf.float32, shape=None)
            self.gamelength = tf.placeholder(tf.int32, shape=None)

            with tf.name_scope("online"):
                self.online_Qvalue, online_var_dictionary = self._DQN(self.state, name="online")

            with tf.name_scope("target"):
                self.target_Qvalue, target_var_dictionary = self._DQN(self.state, name="target")

            with tf.name_scope("copy"):
                # 온라인 네트워크의 가중치를 타깃 네트워크로 복사하기
                self.cpFromOnlinetoTarget = [target_var.assign(online_var_dictionary[var_name]) for var_name, target_var
                                             in target_var_dictionary.items()]

            with tf.name_scope("update_variable"):
                trainable_var_list = tf.global_variables()

            with tf.name_scope("saver"):
                self.saver = tf.train.Saver(var_list=trainable_var_list, max_to_keep=5)

            with tf.name_scope("Loss"):
                Qvalue = tf.reduce_sum(
                    tf.multiply(self.online_Qvalue, tf.one_hot(self.action, self._action_space_number)),
                    axis=1,
                    keepdims=True)

                error = tf.abs(self.target - Qvalue)
                # 0 < error < 1 일 때는 tf.square(clipped_error)
                # error > 1 일때는 2*error-1 적용 - 선형
                clipped_error = tf.clip_by_value(error, 0, 1)
                linear_error = 2 * (error - clipped_error)
                self.loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)

            with tf.name_scope("trainer"):
                optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, momentum=self.momentum)
                self.train_operation = optimizer.minimize(self.loss, var_list=trainable_var_list)
                all_var_list = tf.global_variables()

            with tf.name_scope("game_infomation"):
                tf.summary.scalar("Loss", self.loss)
                tf.summary.scalar("Reward", self.rewards)
                tf.summary.scalar("Qvalue", self.Qvalues)
                tf.summary.scalar("Game length", self.gamelength)

            self.summary_operation = tf.summary.merge_all()

            for operator in (self.state, self.online_Qvalue):
                tf.add_to_collection("way", operator)

            # generator graph 구조를 파일에 쓴다.
            meta_save_file_path = os.path.join(self.model_name, 'Graph.meta')
            self.saver.export_meta_graph(meta_save_file_path, collection_list=["way"])

            if self.only_draw_graph:
                print('<<< Graph.meta 파일만 저장하고 종료합니다. >>>')
                exit(0)

        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        # config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.1
        self.sess = tf.Session(graph=self.Graph, config=config)

        # 가중치 초기화 및 online DQN -> target DQN으로 복사 or 복구
        print("<<< initializing!!! >>>")
        self.sess.run(tf.variables_initializer(all_var_list))
        self.sess.run(self.cpFromOnlinetoTarget)

        ckpt = tf.train.get_checkpoint_state(self.model_name)
        if (ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path)):
            print("<<< all variable retored except for optimizer parameter >>>")
            print("<<< Restore {} checkpoint!!! >>>".format(os.path.basename(ckpt.model_checkpoint_path)))
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            self.start = int(os.path.basename(ckpt.model_checkpoint_path).split("-")[-1])
        else:
            self.start = 1

    def _normalizaiton(self, value=None, factor=255.0):
        return np.divide(value, factor)

    @property
    def train(self):

        if os.path.exists("tensorboard/{}".format(self.model_name)):
            shutil.rmtree("tensorboard/{}".format(self.model_name))
        self.summary_writer = tf.summary.FileWriter(os.path.join("tensorboard", self.model_name), self.sess.graph)

        update_counter = 0  # 학습 횟수 카운터!!!
        gamelength = 0
        totalgame = 0
        totalQvalues = 0
        totalrewards = 0
        gamestate = True  # 게임 초기화 및 게임의 완료 정보를 위함

        # 실질적으로 (self.training_step - self.rememorystackNum) 만큼만 학습한다.
        for step in tqdm(range(self.start, self.training_start_point + self.training_step + 1, 1)):

            if step % self.display_step == 0 and self.display:
                print("\n<<< Validation at {} step >>>".format(step))
                val_step = 0
                valid_total_reward = 0
                valid_sequence_state = self._concatenated_state(self.val_env.reset())

                # 왜 100으로 제한을 뒀는가? while True로 하면... break 못하는 현상이 발생한다.
                for _ in range(100):
                    val_step += 1
                    time.sleep(1 / 30)  # 30fps

                    self.val_env.render()
                    valid_action = self.sess.run(self.online_Qvalue,
                                                 feed_dict={self.state: self._normalizaiton([valid_sequence_state])})
                    valid_next_state, valid_reward, valid_gamestate, _ = self.val_env.step(np.argmax(valid_action))
                    valid_sequence_state = np.concatenate((valid_sequence_state[:, :, 1:],
                                                           self._data_preprocessing(valid_next_state)[:, :,
                                                           np.newaxis]), axis=-1)
                    valid_total_reward += valid_reward

                    # 점수를 받은 부분만 표시하기
                    if valid_reward != 0:
                        print("게임 step {} -> reward :{}".format(val_step, valid_reward))
                    if valid_gamestate:
                        print("total reward : {}\n".format(valid_total_reward))
                        break

                self.val_env.close()

            # self.obs 자체가 self.RM.append 됨에 따라 한칸씩 밀린다.
            if gamestate:
                totalgame += 1
                # 현재의 연속된 관측을 연결하기 -> 84 x 84 x self.frame_size ,
                self.sequence_state = self._concatenated_state(self.env.reset())

            # 온라인 DQN을 시작한다. / 왜 normalization을 하는 것인가? scale을 맞춰주는것!!!
            online_Qvalue = self.sess.run(self.online_Qvalue,
                                          feed_dict={self.state: self._normalizaiton([self.sequence_state])})
            action = self._epsilon_greedy(online_Qvalue, step)
            next_state, reward, gamestate, _ = self.env.step(action)

            # # reward -1, 0, 1로 제한하기
            reward = np.clip(reward, a_min=-1, a_max=1)
            ''' 
            재현 메모리 실행
            why ? not gamestate -> 1게임이 끝나면, gamestate 는 True(즉, 1)를 반환하는데,
            이는 게임 종료에 해당함으로, gamestate가 0(즉 not True = False = 0)이 되어야 학습할 때 target_Qvalue를 0으로 만들 수 있다.

            '''
            # 가장 중요한 부분이라고 생각한다. 시간의 흐름을 저장하는 곳!!!
            self.RM.remember(self.sequence_state, action, reward, self._data_preprocessing(next_state), not gamestate)
            self.sequence_state = self.RM.next_sequence_state

            # 1게임이 얼마나 지속? gamelength, 1게임의 q 가치의 평균 값
            totalrewards += reward
            gamelength += 1
            totalQvalues += (np.max(online_Qvalue) / gamelength)

            if step < self.training_start_point or step % self.training_interval != 0:
                continue

            ##################################### 학습 #########################################

            sampled_state, sampled_action, sampled_reward, sampled_next_state, continues = self._sample_memories

            target_Qvalue = self.sess.run(self.target_Qvalue,
                                          feed_dict={self.state: self._normalizaiton(sampled_next_state)})
            target_Qvalue = sampled_reward + continues * self.discount_factor * np.max(target_Qvalue, axis=1,
                                                                                       keepdims=True)
            _, loss = self.sess.run([self.train_operation, self.loss],
                                    feed_dict={self.state: self._normalizaiton(sampled_state),
                                               self.action: sampled_action,
                                               self.target: target_Qvalue})

            update_counter += 1
            # self.copy_step마다 online DQN -> target DQN으로 복사
            if update_counter % self.copy_step == 0:
                print("<<< '{}'번째 'online copy' to 'target' >>>".format(update_counter // self.copy_step))
                self.sess.run(self.cpFromOnlinetoTarget)

            # self.save_step 마다 tensorboard 및 가중치 저장 :
            if update_counter % self.save_step == 0:

                # 학습 과정은 Tensorboard에서 확인하자
                summary_str = self.sess.run(self.summary_operation,
                                            feed_dict={self.state: self._normalizaiton(sampled_state),
                                                       self.action: sampled_action,
                                                       self.target: target_Qvalue, self.rewards: totalrewards,
                                                       self.gamelength: gamelength, self.Qvalues: totalQvalues})
                self.summary_writer.add_summary(summary_str, global_step=update_counter)

                if not os.path.exists(self.model_name):
                    os.makedirs(self.model_name)
                self.saver.save(self.sess, self.model_name + "/", global_step=update_counter,
                                write_meta_graph=False)

            if gamestate:
                totalQvalues = 0
                gamelength = 0
                totalrewards = 0

        print("<<< 학습간 전체 게임 횟수 : {} >>>".format(totalgame))
        # 닫기
        self.sess.close()
        self.env.close()

    @property
    def test(self):
        tf.reset_default_graph()
        meta_path = glob.glob(os.path.join(self.model_name, '*.meta'))
        if len(meta_path) == 0:
            print("<<< Graph가 존재 하지 않습니다. 그래프를 그려 주세요. - only_draw_graph = True >>>")
            print("<<< 강제 종료 합니다. >>>")
            exit(0)
        else:
            print("<<< Graph가 존재 합니다. >>>")

        Graph = tf.Graph()
        with Graph.as_default():

            saver = tf.train.import_meta_graph(meta_path[0], clear_devices=True)  # meta graph 읽어오기
            if saver == None:
                print("<<< meta 파일을 읽을 수 없습니다. >>>")
                print("<<< 강제 종료합니다. >>>")
                exit(0)

            state, online_Qvalue = tf.get_collection('way')

        with tf.Session(graph=Graph) as sess:
            ckpt = tf.train.get_checkpoint_state(self.model_name)
            if ckpt == None:
                print("<<< checkpoint file does not exist>>>")
                print("<<< Exit the program >>>")
                exit(0)

            if (ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path)):
                print("<<< all variable retored except for optimizer parameter >>>")
                print("<<< Restore {} checkpoint!!! >>>".format(os.path.basename(ckpt.model_checkpoint_path)))
                saver.restore(sess, ckpt.model_checkpoint_path)

            step = 0
            total_reward = 0
            frames = []
            sequence_state = self._concatenated_state(self.env.reset())

            while True:

                step += 1
                time.sleep(1 / 30)  # 30fps
                self.env.render()

                frame = self.env.render(mode="rgb_array")
                frames.append(frame)

                action = sess.run(online_Qvalue, feed_dict={state: self._normalizaiton([sequence_state])})
                next_state, reward, gamestate, _ = self.env.step(np.argmax(action))
                total_reward += reward

                # 예전의 가장 왼쪽의 프레임 날려버리기
                sequence_state = np.concatenate(
                    (sequence_state[:, :, 1:], self._data_preprocessing(next_state)[:, :, np.newaxis]),
                    axis=-1)

                if reward != 0:
                    print("게임 step {} -> reward :{}".format(step, reward))

                if gamestate:
                    print("total reward : {}".format(total_reward))
                    break

            self.env.close()

            if self.SaveGameMovie:
                # 애니매이션 만들기
                fig = plt.figure(figsize=(6, 8))
                patch = plt.imshow(frames[0])  # 첫번째 scene 보여주기
                plt.axis("off")  # 축 제거
                ani = animation.FuncAnimation(fig,
                                              func=lambda i, frames, patch: patch.set_data(frames[i]),
                                              fargs=(frames, patch),
                                              frames=len(frames),
                                              repeat=True)

                # sudo apt-get install ffmepg 를 하시고 ffmpeg를 사용하기
                ani.save("{}.mp4".format(self.model_name), writer="ffmpeg", fps=30, dpi=100)
                # ani.save("{}.gif".format(self.model_name), writer="imagemagick", fps=30, dpi=100) # 오류 발생함.. 이유는? 모
                plt.show()


if __name__ == "__main__":
    Atari = model(
        # https://gym.openai.com/envs/#atari
        # ex) TennisDeterministic-v0, PongDeterministic-v4, BattleZoneDeterministic-v4, BreakoutDeterministic-v4
        model_name="BreakoutDeterministic-v4",
        training_display=(True, 100000),
        training_step=200000000,
        training_start_point=10000,
        # 4번마다 한번씩만 학습 하겠다는 것이다.
        # -> 4번중 3번은 게임을 진행해보고 4번째에는 그 결과들을 바탕으로 학습을 하겠다는 이야기
        training_interval=4,
        rememorystackNum=500000,
        save_step=10000,  # 가중치 업데이트의 save_step 마다 저장한다.
        copy_step=10000,  # 가중치 업데이트의 copy_step 마다 저장한다.
        framesize=4,  # 입력 상태 개수
        learning_rate=0.00025,
        momentum=0.95,
        egreedy_max=1,
        egreedy_min=0.1,
        egreedy_step=1000000,
        discount_factor=0.99,
        batch_size=32,
        with_replacement=True,  # True : 중복추출, False : 비중복 추출
        only_draw_graph=False,  # model 초기화 하고 연산 그래프 그리기
        SaveGameMovie=True)

    Atari.train  # 학습 하기
    Atari.test  # 테스트 하기
