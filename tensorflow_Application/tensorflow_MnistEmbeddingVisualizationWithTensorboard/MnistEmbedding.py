import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.examples.tutorials.mnist import input_data

''' 참고
https://www.tensorflow.org/guide/embedding
http://www.pinchofintelligence.com/simple-introduction-to-tensorboard-embedding-visualisation/
'''


def create_sprite_image(images):
    '''아래의 규칙을 따라 만들어야 한다.
    To use images as metadata, you must produce a single sprite image,
    consisting of small thumbnails, one for each vector in the embedding.
    The sprite should store thumbnails in row-first order:
    the first data point placed in the top left and the last data point in the bottom right,
    though the last row doesn't have to be filled, as shown below.

    image sprites? - An image sprite is a collection of images put into a single image.
    - 이미지 스프라이트는 단일 이미지에 들어있는 이미지 모음!!!
    '''

    images = images.reshape((-1, 28, 28))
    height = images.shape[1]
    width = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    spriteimage = np.zeros((height * n_plots, width * n_plots))

    for h in range(0, n_plots, 1):
        for w in range(0, n_plots, 1):
            this_filter = h * n_plots + w
            if this_filter < images.shape[0]:
                spriteimage[h * height:(h + 1) * height,
                w * width:(w + 1) * width] = images[this_filter]

    return 1 - spriteimage  # 색반전(배경이 하얀색이게!!!)


def model(embedding_count=500):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    LOG_DIR = 'TensorboardandWeight'  # tensorboard와 weight가 저장될 경로 - embedding을 위해서 둘의 경로는 같아야한다.
    path_for_mnist_sprites = os.path.join(LOG_DIR, 'mnistdigits.png')  # 여기에 이미지를 저장해 줘야한다.
    path_for_mnist_metadata = os.path.join(LOG_DIR, 'metadata.tsv')

    if os.path.exists(LOG_DIR):
        shutil.rmtree(LOG_DIR)

    # print(tf.get_default_graph()) #기본그래프이다.
    JG_Graph = tf.Graph()  # 내 그래프로 설정한다.- 혹시라도 나중에 여러 그래프를 사용할 경우를 대비
    with JG_Graph.as_default():  # as_default()는 JG_Graph를 기본그래프로 설정한다.

        # MNIST 데이터 불러오기
        batch_xs, batch_ys = mnist.train.next_batch(embedding_count)

        # 요 아래의 embedding_var 변수에  위의 batch_xs를 넣어주자
        embedding_var = tf.Variable(batch_xs, name="MNIST_Embedding")

        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)

        # 1. TSV(tab seperated file) 파일 쓰기 
        with open(path_for_mnist_metadata, 'w') as f:
            f.write("Index\tLabel\n")
            for index, label in enumerate(batch_ys):
                f.write("{}\t{}\n".format(index, label))

        # 2. sprite 파일 만들기
        to_visualise = batch_xs
        sprite_image = create_sprite_image(to_visualise)
        plt.imsave(path_for_mnist_sprites, sprite_image, cmap='gray')

        # embedding 하기 위해 해야하는 작업
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name

        # 임베딩 벡터를 연관 레이블 or 이미지와 연결
        embedding.metadata_path = os.path.abspath(path_for_mnist_metadata)  # 'metadata.tsv'

        # sprite 파일을 찾기 - 아래의 코드를 추가해주지 않으면, 이미지가 아니라 단순 점으로 보인다.
        # 물론 'mnistdigits.png' 파일을 미리 만들어 둬야 한다.
        embedding.sprite.image_path = os.path.abspath(path_for_mnist_sprites)  # 'mnistdigits.png'
        embedding.sprite.single_image_dim.extend([28, 28])

        saver = tf.train.Saver()

        with tf.Session() as sess:

            # tensorboard FileWriter 선언
            summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
            # tensorboard에 config(임베딩 정보) 추가해서 보여주는 역할
            projector.visualize_embeddings(summary_writer, config)

            sess.run(tf.global_variables_initializer())
            saver.save(sess, os.path.join(LOG_DIR, "MNIST_Embedding.ckpt"))
