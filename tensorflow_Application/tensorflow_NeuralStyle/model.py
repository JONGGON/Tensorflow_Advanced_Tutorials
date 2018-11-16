import os
import shutil
import urllib

import cv2
from tqdm import *

import data_preprocessing as dp
from VGG import *


def neuralstyle(model_file_path="", epoch=None, show_period=None, optimizer_selection="adam", learning_rate=None, \
                image_size=None, \
                content_image=None, style_image=None, content_a=None, style_b=None, initial_noise_image=None):
    if os.path.exists("tensorboard"):
        shutil.rmtree("tensorboard");

    # layer selection
    CONTENT_LAYERS = 'relu4_2'
    STYLE_LAYERS = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
    weighting_factors = np.divide(np.ones(shape=np.shape(STYLE_LAYERS)), len(STYLE_LAYERS))

    def artistic_Image(image, count=None, show=False):

        mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
        std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))

        image = image[0]
        image = (image * std) + mean
        image = np.clip(image, a_min=0, a_max=1) * 255
        r, g, b = cv2.split(image)
        image = cv2.merge([b, g, r])
        image = np.clip(image, a_min=0, a_max=255).astype('uint8')

        if not os.path.exists("article"):
            os.makedirs("article")

        if count != None:
            cv2.imwrite("article/artistic_Image_{}.png".format(count), image)

        if show:
            cv2.imwrite("article/artistic_Image_final.png", image)
            img = cv2.imread("article/artistic_Image_final.png", cv2.IMREAD_COLOR)
            cv2.imshow("artistic_Image", img)
            esc = cv2.waitKey(0)
            if esc == 27:  # esc key
                cv2.destroyAllWindows()

    def Algorithm(vgg_content, vgg_style, vgg_noise):

        # (1)compute content lose
        # using conv4_2
        _, height, width, filter = vgg_content[CONTENT_LAYERS].get_shape()
        n = vgg_noise[CONTENT_LAYERS]
        c = vgg_content[CONTENT_LAYERS]
        n = tf.reshape(n, shape=(-1, height.value * width.value))
        c = tf.reshape(c, shape=(-1, height.value * width.value))
        c_loss = tf.reduce_mean(tf.divide(tf.square(n - c), 2))
        c_loss = tf.multiply(c_loss, content_a)

        # (1)compute style lose
        # using cov1_1 ,cov2_1 ,cov3_1 ,cov4_1 ,cov5_1
        s_loss = 0
        for i, SL in enumerate(STYLE_LAYERS):
            _, height, width, filter = vgg_style[SL].get_shape()
            n = vgg_noise[SL]
            s = vgg_style[SL]

            N = filter.value
            M = height.value * width.value

            n = tf.reshape(n, shape=(-1, M))
            s = tf.reshape(s, shape=(-1, M))
            # gram_matrix
            gram_n = tf.matmul(n, n, transpose_a=False, transpose_b=True)  # (filter, filter)
            gram_s = tf.matmul(s, s, transpose_a=False, transpose_b=True)  # (filter, filter)
            s_loss = s_loss + tf.reduce_mean(
                tf.multiply(tf.divide(tf.square(gram_n - gram_s), 4 * M * N), weighting_factors[i] * 2))

        s_loss = tf.multiply(s_loss, style_b)
        loss = c_loss + s_loss
        return loss, c_loss, s_loss

    def Trainer(cost, global_step):

        tf.summary.scalar("cost", cost)
        if optimizer_selection == "Adam":
            print("using {} optimizer".format(optimizer_selection))
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif optimizer_selection == "RMSP":
            print("using {} optimizer".format(optimizer_selection))
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        else:
            print("using {} optimizer".format(optimizer_selection))
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            # global_step: Optional `Variable` to increment by one after the variables have been updated.
        train_operation = optimizer.minimize(cost, global_step=global_step)
        return train_operation

    # print(tf.get_default_graph()) #기본그래프이다.
    JG_Graph = tf.Graph()  # 내 그래프로 설정한다.- 혹시라도 나중에 여러 그래프를 사용할 경우를 대비
    with JG_Graph.as_default():  # as_default()는 JG_Graph를 기본그래프로 설정한다.
        # 1. Data Preprocessing and noise data
        content_img, style_img, noise_image = dp.data_preprocessing(content_image=content_image, \
                                                                    style_image=style_image, image_size=image_size)
        # initializing noise image below values
        if initial_noise_image == "content_image":
            noise_image = content_img
        elif initial_noise_image == "style_image":
            noise_image = style_img

        # 1. variable definition
        with tf.variable_scope("noise_shared_variables") as scope:
            noise_variable = tf.get_variable("noise_image", initializer=noise_image.astype(np.float32),
                                             dtype=tf.float32)
            # or scope.reuse_variables()

        with tf.name_scope("content"):
            content_placeholder = tf.placeholder(tf.float32, shape=content_img.shape)
        with tf.name_scope("style"):
            style_placeholder = tf.placeholder(tf.float32, shape=style_img.shape)

        # download URL : http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat
        if os.path.exists(model_file_path):
            print("vgg19.params exists")
        else:
            print("vgg19.params downloading")
            url = "http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat"
            urllib.request.urlretrieve(url, model_file_path)
            print("vgg19.params downloading completed")

        vgg19 = VGG19(model_file_path)

        # 2. Algorithm
        with tf.name_scope("content_inference"):
            vgg_content = vgg19(content_placeholder)
        with tf.name_scope("style_inference"):
            vgg_style = vgg19(style_placeholder)
        with tf.name_scope("noise_inference"):
            vgg_noise = vgg19(noise_variable)
        with tf.name_scope("Neural_Style_loss"):
            loss, c_loss, s_loss = Algorithm(vgg_content, vgg_style, vgg_noise)

        # 3. optimizer
        with tf.name_scope("trainer"):
            global_step = tf.Variable(0, name="global_step", trainable=False, dtype=tf.int32)
            train_operation = Trainer(loss, global_step)

        with tf.name_scope("tensorboard"):
            tf.summary.image("noise_image", noise_variable, max_outputs=3)
            tf.summary.image("content_image", content_placeholder, max_outputs=3)
            tf.summary.image("style_image", style_placeholder, max_outputs=3)
            summary_operation = tf.summary.merge_all()

    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(graph=JG_Graph, config=config) as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter("tensorboard", sess.graph)

        for i in tqdm(range(1, epoch + 1, 1)):
            feed_dict = {content_placeholder: content_img, style_placeholder: style_img}
            _, tl, cl, sl = sess.run([train_operation, loss, c_loss, s_loss], feed_dict=feed_dict)
            print("epoch : {} / total cost : {}, content loss : {}, style loss : {}".format(i, tl, cl, sl))
            summary_str = sess.run(summary_operation, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=sess.run(global_step))

            # saving image
            if (i + 1) % show_period == 0:
                artistic_Image(sess.run(noise_variable), count=i + 1, show=False)
        print("Optimization Finished!")

        artistic_Image(sess.run(noise_variable), show=True)


if __name__ == "__main__":

    print("Neural Style Implementation")
    # download URL : http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat
    if not os.path.exists("weights"):
        os.makedirs("weights")
    model_file_path = 'weights/imagenet-vgg-verydeep-19.mat'

    # content_a  / style_b = 1/1000
    content_image = "content/person.jpg"
    style_image = "style/seated-nude.jpg"
    initial_noise_image = "content_image"  # or style image or noise -> Assigning an initial value to the content image is faster than assigning noise.

    # image_size height , width -> is expected to be at least 224.
    # optimizer_selection -> Adam, RMSP, SGD
    neuralstyle(model_file_path=model_file_path, epoch=500, show_period=100, optimizer_selection="Adam", \
                learning_rate=0.1,
                image_size=(380, 683), \
                content_image=content_image, style_image=style_image, content_a=1, style_b=1000, \
                initial_noise_image=initial_noise_image)
else:
    print("Neural Style imported")
