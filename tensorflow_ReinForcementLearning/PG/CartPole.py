import glob
import os

import gym
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import *

# 핸즈온 머신러닝 책의 Cartpole 예제 참고 및 수정
'''
정책 파라미터에 대한 보상의 그라디언트를 평가해서 높은 보상의 방향을 따르는
그라디언트로('경사 상승법') 파라미터를 수정하는 최적화 기법을 사용 - Policy Gradient(정책 그래디언트)

PG 알고리즘 : 높은 보상을 얻는 방향의 그라디언트로 정책의 파라미터를 최적화하는 알고리즘
 - 로날드 윌리엄스의 REINFOCE 알고리즘이 유명함
 - 일단 에피소드(게임)를 몇 번 진행해보고 이를 평균 내어 학습하기 때문에 몬테카를로 정책 그래디언트(Monte Carlo Policy Gradient)라고도 함
'''


class CartPole(object):

    def __init__(self, model_name="CartPole", epoch=1000, gradient_update=10, learning_rate=0.01, training_display=True, SaveGameMovie=True,
                 discount_factor=0.95, save_weight=100, save_path="CartPole", only_draw_graph=False):

        self.env = gym.make("CartPole-v1")
        self.model_name = model_name
        self.n_hidden = 80
        self.n_input = 4
        self.n_output = 1

        self.epoch = epoch  # 학습 횟수
        self.training_display = training_display
        self.SaveGameMovie = SaveGameMovie
        self.learning_rate = learning_rate  # 학습률
        self.gradient_update = gradient_update  # 10번의 게임이 끝난 후 정책을 훈련한다
        self.save_weight = save_weight  # 10번의 게임이 끝날때마다 모델을 저장한다.
        self.discount_factor = discount_factor  # 할인 계수
        self.save_path = save_path  # 가중치가 저장될 경로
        self.only_draw_graph = only_draw_graph

        # Policy Gradient 연산 그래프 그리기
        self._build_Graph()

    # 행동 평가 : 신용 할당 문제 -> 할인 계수 도입
    def _discount_rewards(self, rewards, discount_factor):
        discount_rewards = np.empty(len(rewards))
        cumulative_rewards = 0
        for step in reversed(range(len(rewards))):
            cumulative_rewards = rewards[step] + cumulative_rewards * discount_factor
            discount_rewards[step] = cumulative_rewards
        return discount_rewards

    # 행동 평가 : 신용 할당 문제 -> 할인 계수 도입
    def _disconut_and_normalize_rewars(self, all_rewards, discount_factor):
        all_discounted_rewards = [self._discount_rewards(rewards, discount_factor) for rewards in all_rewards]
        flat_rewards = np.concatenate(all_discounted_rewards, axis=0)
        reward_mean = flat_rewards.mean()
        reward_std = flat_rewards.std()
        return [(discounted_rewards - reward_mean) / reward_std for discounted_rewards in all_discounted_rewards]

    def _build_Graph(self):

        Graph = tf.Graph()  # 내 그래프로 설정한다.- 혹시라도 나중에 여러 그래프를 사용할 경우를 대비
        with Graph.as_default():  # as_default()는 JG_Graph를 기본그래프로 설정한다.

            with tf.name_scope("Network"):

                self.initializer = tf.contrib.layers.variance_scaling_initializer()  # He 초기화
                self.x = tf.placeholder(tf.float32, shape=(None, self.n_input))
                hidden = tf.layers.dense(self.x, self.n_hidden, activation=tf.nn.elu,
                                         kernel_initializer=self.initializer)
                logits = tf.layers.dense(hidden, self.n_output, kernel_initializer=self.initializer)
                outputs = tf.nn.sigmoid(logits)

                '''
                행동 선택
                output = 1 -> action : 0
                output = 0 -> action : 1
                '''
                p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs])
                self.action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)
                y = 1. - tf.to_float(self.action)

            with tf.name_scope("update_variable"):
                trainable_var_list = tf.global_variables()

            with tf.name_scope("saver"):
                self.saver = tf.train.Saver(var_list=trainable_var_list, max_to_keep=5)

            '''
            log(p)를 커지는 방향으로 그라디언트를 업데이트 해야하므로
            sigmoid_cross_entropy 를 최소화 하는 것과 같다.
            '''
            with tf.name_scope("Loss"):
                self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)

            with tf.name_scope("trainer"):
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                grads_and_vars = optimizer.compute_gradients(self.cross_entropy)

                # <<< credit assignment problem(신용 할당 문제) 을 위한 작업 >>> #
                self.gradients = [grad for grad, variable in grads_and_vars]
                self.gradient_placeholders = []
                grads_and_vars_feed = []

                for grad, variable in grads_and_vars:
                    # credit assignment problem(신용 할당 문제)를 위해 담아둘공간이 필요해서 아래와 같은 작업 진행
                    gradient_placeholder = tf.placeholder(tf.float32, shape=None)
                    self.gradient_placeholders.append(gradient_placeholder)

                    grads_and_vars_feed.append((gradient_placeholder, variable))
                self.training_op = optimizer.apply_gradients(grads_and_vars_feed)
                all_var_list = tf.global_variables()

            for operator in (self.x, self.action):
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

        self.sess = tf.Session(graph=Graph, config=config)

        print("<<< initializing!!! >>>")
        self.sess.run(tf.variables_initializer(all_var_list))
        ckpt = tf.train.get_checkpoint_state(self.model_name)

        if (ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path)):
            print("<<< all variable retored except for optimizer parameter >>>")
            print("<<< Restore {} checkpoint!!! >>>".format(os.path.basename(ckpt.model_checkpoint_path)))
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            self.start = int(os.path.basename(ckpt.model_checkpoint_path).split("-")[-1])
        else:
            self.start = 1

    @property
    def train(self):

        for epoch in tqdm(range(1, self.epoch + 1, 1)):
            all_rewards = []
            all_gradients = []
            total_reward = 0
            for _  in range(self.gradient_update):
                current_rewards = []
                current_gradients = []
                obs = self.env.reset()
                while True:

                    if self.training_display:
                        self.env.render()

                    # action = tf.multinomial(tf.log(p_left_and_right), num_samples=1) 가 2차원 배열로 반환!!!
                    action_val, gradients_val = self.sess.run([self.action, self.gradients],
                                                              feed_dict={self.x: obs.reshape(1, self.n_input)})
                    obs, reward, done, info = self.env.step(action_val[0][0])

                    total_reward += reward
                    # 모든 행동을 다 고려하다니.. 오래 걸릴 수 밖에 없구나...
                    current_rewards.append(reward)
                    current_gradients.append(gradients_val)
                    if done:
                        print("<<< 보상 : {}>>>".format(total_reward))
                        total_reward = 0
                        break
                # self.update_periods(ex) 10 게임) 마다 보상과 그라디언트를 append 한다.
                all_rewards.append(current_rewards)
                all_gradients.append(current_gradients)

            ''' 
            정규화된 결과가 나온다. 왜 정규화를 해야하나?
            책에는? -> 행동에 대해 신뢰할만한 점수를 얻으려면 많은 에피소드(게임)를 실행하고
            모든 행동의 점수를 정규화 해야한다고 나와있다.
            '''
            all_rewards = self._disconut_and_normalize_rewars(all_rewards, self.discount_factor)
            feed_dict = {}

            for var_index, gradient_placeholder in enumerate(self.gradient_placeholders):
                # 모든 에피소드와 모든 스텝에 걸쳐 그라디언트와 보상점수를 곱한다. -> 이후 각 그라디언트에 대해 평균을 구한다.
                # 이래서 오래 걸린다...
                mean_gradients = np.mean(
                    [reward * all_gradients[game_index][step][var_index] for game_index, rewards in
                     enumerate(all_rewards) for step, reward in enumerate(rewards)], axis=0)

                feed_dict[gradient_placeholder] = mean_gradients
            # 평균 그라디언트를 훈련되는 변수마다 하나씩 주입하여 훈련연산을 실행한다.
            self.sess.run(self.training_op, feed_dict=feed_dict)

            if epoch % self.save_weight == 0:
                self.saver.save(self.sess, self.save_path + "/Cartpole.ckpt", global_step=epoch, write_meta_graph=False)

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

            x, action = tf.get_collection('way')

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

            obs = self.env.reset()
            step = 1
            total_reward = 0
            frames = []

            while True:
                frame = self.env.render(mode="rgb_array")
                frames.append(frame)

                action_val = sess.run(action, feed_dict={x: obs.reshape(1, self.n_input)})
                obs, reward, done, _ = self.env.step(action_val[0][0])
                print("게임 step {} -> reward : {}".format(step, reward))

                if done:
                    print("total reward : {}".format(total_reward+1))
                    self.env.close()
                    break

                step += 1
                total_reward += reward

            if self.SaveGameMovie:
                # 그림 그리기
                fig = plt.figure(figsize=(8, 8))
                patch = plt.imshow(frames[0])  # 첫번째 scene 보여주기
                plt.axis("off")  # 축 제거
                ani = animation.FuncAnimation(fig,
                                              func=lambda i, frames, patch: patch.set_data(frames[i]),
                                              fargs=(frames, patch),
                                              frames=len(frames),
                                              repeat=True)
                ani.save("CartPole.mp4", writer=None, fps=30, dpi=100) # "imagemagick", "ffmpeg"
                plt.show()


if __name__ == "__main__":
    CartPole(model_name="CartPole", epoch=1, gradient_update=10, learning_rate=0.01, training_display=True, SaveGameMovie=True,
             discount_factor=0.95, save_weight=100, save_path="CartPole", only_draw_graph=False)
