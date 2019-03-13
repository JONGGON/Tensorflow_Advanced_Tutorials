import os
import sys

path = os.path.dirname(__file__)
#sys.path.append(path)
sys.path.insert(0, path)
from operation import *

# or
# python default 설정에는 같은 디렉토리나, sys,path의 파일만 import 할 수 있다.
# 같은 폴더, 하위 폴더 내에서는
#from .operation import * # 이방법도 있다.

class Network(object):

    def __init__(self, class_number, init_filter_size, regularizer=None, scale=0.000001, keep_probability = 0.5):

        self.class_number = class_number
        self.init_filter_size = init_filter_size
        self.regularizer  = regularizer
        self.scale = scale
        self.keep_probability = keep_probability

    def UNET(self, images=None, name=None):

        conv_list = []
        # 입력 256x256 기준
        with tf.variable_scope(name):
            with tf.variable_scope("encoder"):
                with tf.variable_scope("conv1"):
                    conv1 = only_conv2d(images, weight_shape=(4, 4, 3, self.init_filter_size),
                                        strides=[1, 2, 2, 1], padding="SAME")
                    # result shape = (batch_size, 128, 128, 32)
                    conv_list.append(conv1)
                with tf.variable_scope("conv2"):
                    conv2 = conv2d(tf.nn.leaky_relu(conv1, alpha=0.2),
                                   weight_shape=(4, 4, self.init_filter_size, self.init_filter_size * 2),
                                   strides=[1, 2, 2, 1], padding="SAME")
                    # result shape = (batch_size, 64, 64, 64)
                    conv_list.append(conv2)
                with tf.variable_scope("conv3"):
                    conv3 = conv2d(tf.nn.leaky_relu(conv2, alpha=0.2),
                                   weight_shape=(4, 4, self.init_filter_size * 2, self.init_filter_size * 4),
                                   strides=[1, 2, 2, 1], padding="SAME")
                    # result shape = (batch_size, 32, 32, 128)
                    conv_list.append(conv3)
                with tf.variable_scope("conv4"):
                    conv4 = conv2d(tf.nn.leaky_relu(conv3, alpha=0.2),
                                   weight_shape=(4, 4, self.init_filter_size * 4, self.init_filter_size * 8),
                                   strides=[1, 2, 2, 1], padding="SAME")
                    # result shape = (batch_size, 16, 16, 256)
                    conv_list.append(conv4)
                with tf.variable_scope("conv5"):
                    conv5 = conv2d(tf.nn.leaky_relu(conv4, alpha=0.2),
                                   weight_shape=(4, 4, self.init_filter_size * 8, self.init_filter_size * 8),
                                   strides=[1, 2, 2, 1], padding="SAME")
                    # result shape = (batch_size, 8, 8, 256)
                    conv_list.append(conv5)
                with tf.variable_scope("conv6"):
                    conv6 = conv2d(tf.nn.leaky_relu(conv5, alpha=0.2),
                                   weight_shape=(4, 4, self.init_filter_size * 8, self.init_filter_size * 8),
                                   strides=[1, 2, 2, 1], padding="SAME")
                    # result shape = (batch_size, 4, 4, 256)
                    conv_list.append(conv6)
                with tf.variable_scope("conv7"):
                    conv7 = conv2d(tf.nn.leaky_relu(conv6, alpha=0.2),
                                   weight_shape=(4, 4, self.init_filter_size * 8, self.init_filter_size * 8),
                                   strides=[1, 2, 2, 1], padding="SAME")
                    # result shape = (batch_size, 2, 2, 256)
                    conv_list.append(conv7)
                with tf.variable_scope("conv8"):
                    conv8 = only_conv2d(tf.nn.leaky_relu(conv7, alpha=0.2),
                                        weight_shape=(4, 4, self.init_filter_size * 8, self.init_filter_size * 8),
                                        strides=[1, 2, 2, 1], padding="SAME")
                    # result shape = (batch_size, 1, 1, 256)
                    conv_list.append(conv8)

            with tf.variable_scope("decoder"):
                with tf.variable_scope("trans_conv1"):
                    trans_conv1 = tf.nn.dropout(
                        conv2d_transpose(tf.nn.relu(conv8), output_shape=tf.shape(conv7),
                                         weight_shape=(4, 4, self.init_filter_size * 8, self.init_filter_size * 8),
                                         strides=[1, 2, 2, 1], padding="SAME"), keep_prob=self.keep_probability)
                    # 주의 : 활성화 함수 들어가기전의 encoder 요소를 concat 해줘야함
                    conv_list.append(trans_conv1)
                    trans_conv1 = tf.concat([trans_conv1, conv7], axis=-1)
                    # result shape = (batch_size, 2, 2, 512)

                with tf.variable_scope("trans_conv2"):
                    trans_conv2 = tf.nn.dropout(
                        conv2d_transpose(tf.nn.relu(trans_conv1), output_shape=tf.shape(conv6),
                                         weight_shape=(4, 4, self.init_filter_size * 8, self.init_filter_size * 16),
                                         strides=[1, 2, 2, 1], padding="SAME"), keep_prob=self.keep_probability)
                    conv_list.append(trans_conv2)
                    trans_conv2 = tf.concat([trans_conv2, conv6], axis=-1)
                    # result shape = (batch_size, 4, 4, 512)

                with tf.variable_scope("trans_conv3"):
                    trans_conv3 = tf.nn.dropout(
                        conv2d_transpose(tf.nn.relu(trans_conv2), output_shape=tf.shape(conv5),
                                         weight_shape=(4, 4, self.init_filter_size * 8, self.init_filter_size * 16),
                                         strides=[1, 2, 2, 1], padding="SAME"), keep_prob=self.keep_probability)
                    conv_list.append(trans_conv3)
                    trans_conv3 = tf.concat([trans_conv3, conv5], axis=-1)
                    # result shape = (batch_size, 8, 8, 512)

                with tf.variable_scope("trans_conv4"):
                    trans_conv4 = conv2d_transpose(tf.nn.relu(trans_conv3), output_shape=tf.shape(conv4),
                                                   weight_shape=(4, 4, self.init_filter_size * 8, self.init_filter_size * 16),
                                                   strides=[1, 2, 2, 1], padding="SAME")
                    conv_list.append(trans_conv4)
                    trans_conv4 = tf.concat([trans_conv4, conv4], axis=-1)
                    # result shape = (batch_size, 16, 16, 512)
                with tf.variable_scope("trans_conv5"):
                    trans_conv5 = conv2d_transpose(tf.nn.relu(trans_conv4), output_shape=tf.shape(conv3),
                                                   weight_shape=(4, 4, self.init_filter_size * 4, self.init_filter_size * 16),
                                                   strides=[1, 2, 2, 1], padding="SAME")
                    conv_list.append(trans_conv5)
                    trans_conv5 = tf.concat([trans_conv5, conv3], axis=-1)
                    # result shape = (batch_size, 32, 32, 256)
                with tf.variable_scope("trans_conv6"):
                    trans_conv6 = conv2d_transpose(tf.nn.relu(trans_conv5), output_shape=tf.shape(conv2),
                                                   weight_shape=(4, 4, self.init_filter_size * 2, self.init_filter_size * 8),
                                                   strides=[1, 2, 2, 1], padding="SAME")
                    conv_list.append(trans_conv6)
                    trans_conv6 = tf.concat([trans_conv6, conv2], axis=-1)
                    # result shape = (batch_size, 64, 64, 128)
                with tf.variable_scope("trans_conv7"):
                    trans_conv7 = conv2d_transpose(tf.nn.relu(trans_conv6), output_shape=tf.shape(conv1),
                                                   weight_shape=(4, 4, self.init_filter_size, self.init_filter_size * 4),
                                                   strides=[1, 2, 2, 1], padding="SAME")
                    conv_list.append(trans_conv7)
                    trans_conv7 = tf.concat([trans_conv7, conv1], axis=-1)
                    # result shape = (batch_size, 128, 128, 64)
                with tf.variable_scope("trans_conv8"):
                    output = only_conv2d_transpose(tf.nn.relu(trans_conv7), output_shape=tf.shape(tf.tile(images[:, :, :,0:1], tf.constant([1, 1, 1, self.class_number]))),
                                                   weight_shape=(4, 4, self.class_number, self.init_filter_size * 2),
                                                   strides=[1, 2, 2, 1], padding="SAME")
                    # result shape = (batch_size, 256, 256, self.class_number)
        return output, conv_list
