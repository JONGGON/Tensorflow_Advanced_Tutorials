import tensorflow as tf

def conv2d(input, weight_shape=None, regularizer=None, scale=0.000001,
           strides=[1, 1, 1, 1], padding="VALID", dilations=[1, 1, 1, 1]):

    # weight_init = tf.contrib.layers.xavier_initializer(uniform=False)
    # weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
    weight_init = tf.contrib.layers.variance_scaling_initializer()

    if regularizer == "L1":
        weight_decay = tf.constant(scale, dtype=tf.float32)
        w = tf.get_variable("w", weight_shape, initializer=weight_init,
                            regularizer=tf.contrib.layers.l1_regularizer(scale=weight_decay))
    elif regularizer == "L2":
        weight_decay = tf.constant(scale, dtype=tf.float32)
        w = tf.get_variable("w", weight_shape, initializer=weight_init,
                            regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay))
    else:
        w = tf.get_variable("w", weight_shape, initializer=weight_init)

    conv_out = tf.nn.conv2d(input, w, strides=strides, padding=padding, dilations=dilations)

    return tf.contrib.layers.instance_norm(conv_out, epsilon=1e-5)

def only_conv2d(input, weight_shape=None, regularizer=None, scale=0.000001,
           strides=[1, 1, 1, 1], padding="VALID", dilations=[1, 1, 1, 1]):

    # weight_init = tf.contrib.layers.xavier_initializer(uniform=False)
    # weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
    weight_init = tf.contrib.layers.variance_scaling_initializer()

    if regularizer == "L1":
        weight_decay = tf.constant(scale, dtype=tf.float32)
        w = tf.get_variable("w", weight_shape, initializer=weight_init,
                            regularizer=tf.contrib.layers.l1_regularizer(scale=weight_decay))
    elif regularizer == "L2":
        weight_decay = tf.constant(scale, dtype=tf.float32)
        w = tf.get_variable("w", weight_shape, initializer=weight_init,
                            regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay))
    else:
        w = tf.get_variable("w", weight_shape, initializer=weight_init)

    conv_out = tf.nn.conv2d(input, w, strides=strides, padding=padding, dilations=dilations)

    return conv_out

def conv2d_transpose(input, output_shape=None, weight_shape=None, regularizer=None, scale=0.000001,
                     strides=[1, 1, 1, 1], padding="VALID", dilations=[1, 1, 1, 1]):

    weight_init = tf.contrib.layers.variance_scaling_initializer()

    if regularizer == "L1":
        weight_decay = tf.constant(scale, dtype=tf.float32)
        w = tf.get_variable("w", weight_shape, initializer=weight_init,
                            regularizer=tf.contrib.layers.l1_regularizer(scale=weight_decay))
    elif regularizer == "L2":
        weight_decay = tf.constant(scale, dtype=tf.float32)
        w = tf.get_variable("w", weight_shape, initializer=weight_init,
                            regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay))
    else:
        w = tf.get_variable("w", weight_shape, initializer=weight_init)

    conv_out = tf.nn.conv2d_transpose(input, w, output_shape=output_shape, strides=strides, padding=padding)

    return tf.contrib.layers.instance_norm(conv_out, epsilon=1e-5)

# batch norm 통과하기 이전의 convolution만 한 결과
def only_conv2d_transpose(input, output_shape=None, weight_shape=None, strides=[1, 1, 1, 1], padding="VALID", regularizer=None, scale=0.000001):

    weight_init = tf.contrib.layers.variance_scaling_initializer()

    if regularizer == "L1":
        weight_decay = tf.constant(scale, dtype=tf.float32)
        w = tf.get_variable("w", weight_shape, initializer=weight_init,
                            regularizer=tf.contrib.layers.l1_regularizer(scale=weight_decay))
    elif regularizer == "L2":
        weight_decay = tf.constant(scale, dtype=tf.float32)
        w = tf.get_variable("w", weight_shape, initializer=weight_init,
                            regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay))
    else:
        w = tf.get_variable("w", weight_shape, initializer=weight_init)

    conv_out = tf.nn.conv2d_transpose(input, w, output_shape=output_shape, strides=strides, padding=padding)
    return conv_out

def partial_conv2d(input, weight_shape, use_bias=True, strides=[1, 1, 1, 1], padding="SAME", pad=0, dilations=[1, 1, 1, 1], regularizer=None,
                   scale=0.000001,scope="partial_conv"):

    with tf.variable_scope(scope):

        if padding == "SAME":

            weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
            slide_window = weight_shape[1] * weight_shape[2] * weight_shape[3]
            if regularizer == "L1":
                weight_decay = tf.constant(scale, dtyp=tf.float32)
                w = tf.get_variable("w", weight_shape, initializer=weight_init,
                                    regularizer=tf.contrib.layers.l1_regularizer(scale=weight_decay))
                # mask weight 는 학습하지 않는다.
                w_mask = tf.get_variable("w_mask", weight_shape, initializer=tf.constant_initializer(1.0),
                                         trainable=False)
            elif regularizer == "L2":
                weight_decay = tf.constant(scale, dtyp=tf.float32)
                w = tf.get_variable("w", weight_shape, initializer=weight_init,
                                    regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay))
                # mask weight 는 학습하지 않는다.
                w_mask = tf.get_variable("w_mask", weight_shape, initializer=tf.constant_initializer(1.0),
                                         trainable=False)
            else:
                w = tf.get_variable("w", weight_shape, initializer=weight_init)
                # mask weight 는 학습하지 않는다.
                w_mask = tf.get_variable("w_mask", weight_shape, initializer=tf.constant_initializer(1.0),
                                         trainable=False)

            # 입력과 같은 크기인 1로 채워진 mask를 만든다.
            mask = tf.ones(shape=tf.shape(input))
            update_mask = tf.nn.conv2d(mask, w_mask, strides=strides, padding=padding)

            # 논문에 있는 내용 -  border를 찾는 부분
            mask_ratio = slide_window / (update_mask + 1e-8)

            '''
            After each partial convolution operation, we then update
            our mask as follows: if the convolution was able to condition
            its output on at least one valid input value, then we
            mark that location to be valid.
            '''
            # 1. case of Big Padding Size - 첫번째 코드
            # update_mask = tf.clip_by_value(update_mask, 0.0, 1.0) # 어차피 0 or 1이라서, 다음과 같은 표현도 가능

            # 2. case of Big Padding Size - 두번째 코드
            update_mask = tf.where(update_mask > 0, tf.ones(tf.shape(update_mask)), tf.zeros(tf.shape(update_mask)))
            mask_ratio = mask_ratio * update_mask

            input = tf.nn.conv2d(input, w, strides=strides, padding=padding)
            output = tf.multiply(input, mask_ratio)

            if use_bias:
                bias_init = tf.constant_initializer(value=0)
                b = tf.get_variable("b", weight_shape[2], initializer=bias_init)
                output = tf.nn.bias_add(output, b)

                # bias를 더해줬으니깐, 1이 아닌 부분이 다시 생기는 것이고, update_mask 를 다시 곱해줌으로써 bias가 더해진 부분을 없애버리기.
                output = tf.multiply(output, update_mask)


            return tf.contrib.layers.instance_norm(output, epsilon=1e-5)

        # 일반 패딩에서는 안되지만, CUDNN과 같이 Padding을 주고  하는 경우, Pad 수를 준다면 가능함
        elif padding == "VALID":
            if pad > 0:
                pass
            else:
                weight_init = tf.contrib.layers.variance_scaling_initializer()
                if regularizer == "L1":
                    weight_decay = tf.constant(scale, dtype=tf.float32)
                    w = tf.get_variable("w", weight_shape, initializer=weight_init,
                                        regularizer=tf.contrib.layers.l1_regularizer(scale=weight_decay))
                elif regularizer == "L2":
                    weight_decay = tf.constant(scale, dtype=tf.float32)
                    w = tf.get_variable("w", weight_shape, initializer=weight_init,
                                        regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay))
                else:
                    w = tf.get_variable("w", weight_shape, initializer=weight_init)

                conv_out = tf.nn.conv2d(input, w, strides=strides, padding=padding, dilations=dilations)

                return tf.contrib.layers.instance_norm(conv_out, epsilon=1e-5)

def group_norm(x, group=2, eps=1e-5, scope='group_norm'):

    with tf.variable_scope(scope):
        Channel = int(x.get_shape()[-1])

        f, b = tf.split(x, group, axis=-1)  # 4채널
        fmean, fvar = tf.nn.moments(f, axes=[1, 2], keep_dims=True)
        bmean, bvar = tf.nn.moments(b, axes=[1, 2], keep_dims=True)
        f = (f - fmean) / tf.sqrt(fvar + eps)
        b = (b - bmean) / tf.sqrt(bvar + eps)

        gamma = tf.get_variable('gamma', [1, 1, 1, Channel], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', [1, 1, 1, Channel], initializer=tf.constant_initializer(0.0))

        x = tf.concat([f, b], axis=-1)
        x = tf.add(tf.multiply(x, gamma), beta)
    return x