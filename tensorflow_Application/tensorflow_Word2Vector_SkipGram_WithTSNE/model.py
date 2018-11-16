import glob
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
from tensorflow.contrib.tensorboard.plugins import projector
from tqdm import tqdm
from data_preprocessing import data_preprocessing

def Word2Vec(TEST=True, tSNE=True, model_name="Word2Vec", weight_selection="encoder",  # encoder or decoder
             vocabulary_size=30000, tSNE_plot=500, similarity_number=8,
             # similarity_number -> 비슷한 문자 출력 개수
             # num_skip : 하나의 문장당 num_skips 개의 데이터를 생성
             validation_number=30, embedding_size=192, batch_size=192, num_skips=8, window_size=4,
             negative_sampling=64, optimizer_selection="SGD", learning_rate=0.1, training_epochs=100,
             display_step=1, weight_sharing=False, *Arg, **kwargs):
    if weight_sharing:
        model_name = "ws" + model_name

    model_name = model_name + '_v' + str(vocabulary_size) + '_e' + str(embedding_size)

    dp = data_preprocessing(url='http://mattmahoney.net/dc/',
                            filename='text8.zip',
                            expected_bytes=31344016,
                            vocabulary_size=vocabulary_size)

    if TEST == False:
        tensorboard_path=glob.glob(os.path.join("model",model_name)+"/events.*")
        if tensorboard_path !=[] and os.path.exists(tensorboard_path[0]):
            os.remove(tensorboard_path[0])

    def embedding_layer(embedding_shape=None, train_inputs=None):
        with tf.variable_scope("embedding"):
            embedding_init = tf.random_uniform(embedding_shape, minval=-1, maxval=1)
            embedding_matrix = tf.get_variable("E", initializer=embedding_init)
            return tf.nn.embedding_lookup(embedding_matrix, train_inputs), embedding_matrix

    def noise_contrastive_loss(weight_shape=None, bias_shape=None, train_labels=None, embed=None, num_sampled=None,
                               num_classes=None, encoder_weight=None, weight_sharing=True):

        with tf.variable_scope("nce"):
            nce_weight_init = tf.truncated_normal(weight_shape, stddev=np.sqrt(1.0 / (weight_shape[1])))
            nce_bias_init = tf.zeros(bias_shape)
            nce_w = tf.get_variable("w", initializer=nce_weight_init)
            nce_b = tf.get_variable("b", initializer=nce_bias_init)

            if weight_sharing:
                total_loss = tf.nn.nce_loss(weights=encoder_weight, biases=nce_b, labels=train_labels, inputs=embed, \
                                            num_sampled=num_sampled, num_classes=num_classes)
                nce_w = encoder_weight
            else:
                total_loss = tf.nn.nce_loss(weights=nce_w, biases=nce_b, labels=train_labels, inputs=embed, \
                                            num_sampled=num_sampled, num_classes=num_classes)

        return tf.reduce_mean(total_loss), nce_w

    def training(cost, global_step):
        tf.summary.scalar("cost", cost)
        '''아래와 같이 API를 사용하여 Learning rate를 줄일수 있는데, global step에 의존적이다.
        API를 사용하지 않고, placeholder or variable or constant로 지정해놓고, 나중에 값을 넣어주는 방법이 더 나은듯?
        '''
        lr = tf.train.exponential_decay(learning_rate=learning_rate, global_step=global_step, decay_steps=50000,
                                        decay_rate=0.99, staircase=True)
        if optimizer_selection == "Adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        elif optimizer_selection == "RMSP":
            optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)
        elif optimizer_selection == "SGD":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        train_operation = optimizer.minimize(cost, global_step=global_step)
        return train_operation

    if not TEST:
        # print(tf.get_default_graph()) #기본그래프이다.
        JG_Graph = tf.Graph()  # 내 그래프로 설정한다.- 혹시라도 나중에 여러 그래프를 사용할 경우를 대비
        with JG_Graph.as_default():  # as_default()는 JG_Graph를 기본그래프로 설정한다.
            with tf.name_scope("feed_dict"):
                train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
                train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

            with tf.name_scope("Skip_Gram"):
                embed, e_matrix = embedding_layer(embedding_shape=(dp.vocabulary_size, embedding_size),
                                                  train_inputs=train_inputs)
                # scope.reuse_variables()

            with tf.name_scope("nce_loss"):
                global_step = tf.Variable(0, name="global_step", trainable=False)
                cost, nce_weight = noise_contrastive_loss(weight_shape=(dp.vocabulary_size, embedding_size), \
                                                          bias_shape=(dp.vocabulary_size), \
                                                          train_labels=train_labels, \
                                                          embed=embed, \
                                                          num_sampled=negative_sampling, \
                                                          num_classes=dp.vocabulary_size, \
                                                          encoder_weight=e_matrix, \
                                                          weight_sharing=weight_sharing)

            # Adam optimizer의 매개변수들을 저장하고 싶지 않다면 여기에 선언해야한다.
            with tf.name_scope("saver"):
                saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=3)

            with tf.name_scope("trainer"):
                train_operation = training(cost, global_step)
                
            summary_operation = tf.summary.merge_all()

            '''
            WHY? 아래 2줄의 코드를 적어 주지 않고, 학습을 하게되면, TEST부분에서 tf.train.import_meta_graph를 사용할 때 오류가 난다. 
            -> 단순히 그래프를 가져오고 가중치를 복원하는 것만으로는 안된다. 세션을 실행할때 인수로 사용할 변수에 대한 
            추가 접근을 제공하지 않기 때문에 아래와 같이 저장을 해놓은 뒤 TEST 시에 불러와서 다시 사용 해야한다. - 그렇지 않으면, JG.get_operations() 함수를
            사용해 출력된 모든 연산의 리스트에서 하나하나 찾아야한다.
            필요한 변수가 있을 시 아래와 같이 추가해서 그래프를 새로 만들어 주면 된다.
            '''
            if weight_sharing:
                tf.add_to_collection("way", e_matrix)
            else:
                for op in (e_matrix, nce_weight):
                    tf.add_to_collection("way", op)

            # generator graph 구조를 파일에 쓴다.
            meta_save_file_path = os.path.join("model", model_name, 'Graph.meta')
            saver.export_meta_graph(meta_save_file_path, collection_list=["way"])

            config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
            config.gpu_options.allow_growth = True

            with tf.Session(graph=JG_Graph, config=config) as sess:
                print("initializing!!!")
                sess.run(tf.global_variables_initializer())
                ckpt = tf.train.get_checkpoint_state(os.path.join("model", model_name))

                if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                    print("Restore {} checkpoint!!!".format(os.path.basename(ckpt.model_checkpoint_path)))
                    saver.restore(sess, ckpt.model_checkpoint_path)

                batches_per_epoch = int(
                    (dp.vocabulary_size * num_skips) / batch_size)  # Number of batches per epoch of training

                '''
                embedding 시각화 할 때 주의할점!!!
                tensorboard가 저장된 위치와 가중치 파라미터의 위치가 같아야된다. 가중치를 불러와서 사용한다.
                '''
                summary_writer = tf.summary.FileWriter(os.path.join("model", model_name), sess.graph)
                with open(os.path.join("model", model_name, "metadata.tsv"), "w") as md:
                    md.write('Word\tIndex\n')
                    for k, v in dp.dictionary.items():
                        md.write("{}\t{}\n".format(k, v)) # word , index

                #임베딩 시각화 하기
                # from tensorflow.contrib.tensorboard.plugins import projector 이 꼭 필요하다!
                config = projector.ProjectorConfig()
                if weight_sharing:
                    embedding_encoder = config.embeddings.add()
                    embedding_encoder.tensor_name = e_matrix.name  # encoder의 weight로만 그린다.
                    embedding_encoder.metadata_path = os.path.abspath(
                        os.path.join("model", model_name, "metadata.tsv"))  # 절대경로를 써주는게 좋음
                else:
                    embedding_encoder = config.embeddings.add()
                    embedding_decoder = config.embeddings.add()
                    embedding_encoder.tensor_name= e_matrix.name # encoder의 weight로만 그린다.
                    embedding_decoder.tensor_name= nce_weight.name # encoder의 weight로만 그린다.
                    # 임베딩 벡터를 연관 레이블 or 이미지와 연결
                    embedding_encoder.metadata_path = os.path.abspath(os.path.join("model", model_name, "metadata.tsv")) #절대경로를 써주는게 좋음
                    embedding_decoder.metadata_path = os.path.abspath(os.path.join("model", model_name, "metadata.tsv")) #절대경로를 써주는게 좋음

                projector.visualize_embeddings(summary_writer, config)

                for epoch in tqdm(range(training_epochs)):
                    avg_cost = 0.
                    for minibatch in range(batches_per_epoch):  # # Number of batches per epoch of training
                        mbatch_x, mbatch_y = dp.generate_batch(batch_size=batch_size, num_skips=num_skips,
                                                               window_size=window_size)
                        feed_dict = {train_inputs: mbatch_x, train_labels: mbatch_y}
                        _, new_cost = sess.run([train_operation, cost], feed_dict=feed_dict)
                        # Compute average loss
                        avg_cost += new_cost / batches_per_epoch
                    print("cost : {0:0.3}".format(avg_cost))

                    if epoch % display_step == 0:
                        summary_str = sess.run(summary_operation, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, global_step=sess.run(global_step))
                        save_path = os.path.join("model", model_name)
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        saver.save(sess, save_path + "/", global_step=sess.run(global_step),
                                   write_meta_graph=False)
                print("Optimization Finished!")

    else:
        tf.reset_default_graph()
        meta_path = glob.glob(os.path.join('model', model_name, '*.meta'))
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
            WHY? 아래 1줄의 코드를 적어 주지 않으면 오류가 발생한다. 
            -> 단순히 그래프를 가져오고 가중치를 복원하는 것만으로는 안된다. 세션을 실행할때 인수로 사용할 변수에 대한 
            추가 접근을 제공하지 않기 때문에 아래와 같이 get_colltection으로 입,출력 변수들을 불러와서 다시 사용 해야 한다.
            '''

            if weight_sharing:
                embedding_matrix = tf.get_collection('way')[0]
            else:
                e_matrix, nce_weight = tf.get_collection('way')
                if weight_selection == "encoder":
                    embedding_matrix = e_matrix
                elif weight_selection == "decoder":
                    embedding_matrix = nce_weight
                else:
                    print("weight_selcetion = 'encoder' or weight_selcetion = 'decoder' 만 가능합니다")
                    print("재 입력 해주세요. 강제 종료합니다.")
                    exit(0)

            # Data preprocessing 에서 가장 많이 출현한 10000개의 단어를 학습데이터로 사용
            # 학습데이터중 앞에서부터 tSNE_plot개의 단어중 validation_numbe개를 무작위로 선택하여 비슷한 단어들을 출력하는 변수.
            validation_inputs = tf.constant(np.random.choice(tSNE_plot, validation_number, replace=False),
                                            dtype=tf.int32)
            norm = tf.sqrt(tf.reduce_sum(tf.square(embedding_matrix), axis=1, keepdims=True))
            normalized = tf.divide(embedding_matrix, norm)
            val_embeddings = tf.nn.embedding_lookup(normalized, validation_inputs)
            '''  
            --> embedding_matrix 가 잘 학습이 되었다면, 비슷한 단어들은 비슷한 벡터의 값을 가질 것이다. - TSNE로 확인 가능하다.
            이 embedding_matrix를 decoder로 출력했을 때, 상위 몇개의 단어는 연관된 의미를 표현할 것이다. 
            - cosine_similarity 은 결국은 decoder의 출력이다 -> shape = (validation_number, vocabulary_size)
            -> 여기서 가장 큰 값이 입력과 같아야 하는 것이고, 최상위 몇개의 값이 가장 큰값(=입력)과 연관이 있는
            단어들을 의미한다고 한다. -> 정말 신기한것 같다.(물론 데이터가 좋아야 한다.)
            '''
            # (-1, embedding_size) x (embedding_size, vocabulary_size)
            cosine_similarity = tf.matmul(val_embeddings, normalized, transpose_b=True)

            with tf.Session(graph=JG) as sess:
                ckpt = tf.train.get_checkpoint_state(os.path.join('model', model_name))

                if ckpt == None:
                    print("<<< checkpoint file does not exist>>>")
                    print("<<< Exit the program >>>")
                    exit(0)

                if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                    print("<<< generator variable retored except for optimizer parameter >>>")
                    print("<<< Restore {} checkpoint!!! >>>".format(os.path.basename(ckpt.model_checkpoint_path)))
                    saver.restore(sess, ckpt.model_checkpoint_path)

                final_embeddings, similarity = sess.run([normalized, cosine_similarity])

                for i in range(validation_number):
                    val_word = dp.reverse_dictionary[sess.run(validation_inputs)[i]]

                    '''argsort()함수는 오름차순으로 similarity(batch_size, vocabulary size) 값을
                    정렬한다. 즉, 가장 높은 확률을 가진 놈은 가장 오른쪽에 있다. 따라서 뒤에서부터 접근하여
                    큰 값들을 가져온다.
                    '''
                    neighbors = (similarity[i, :]).argsort()[
                                -1:-1 - similarity_number - 1:-1]  # -1 ~ -similarity_number-1 까지의 값
                    # neighbors = (-similarity[i, :]).argsort()[0:similarity_number+1]  # -1 ~ -similarity_number-1 까지의 값
                    print_str = "< Nearest neighbor of {} / ".format(val_word)

                    # word2vec도 결국은 autoencoder 이다. 가장 가까운 단어는 자기 자신이 나온다.[index = 0]
                    # 해당 단어와 가장 가까운 것은 자기 자신이기 때문에 학습 하는 과정에서 자연스럽게 정해진다.
                    '''
                    학습이 전혀 되지 않는 상태에서, Test만 해도 0번은 자기 자신이 나오는데 이는
                    embedding_init가 균일분포로 초기화 되었기 때문이다.(tf.random_uniform(embedding_shape, minval=-1, maxval=1))
                    embedding_init가 1이나 0으로 초기화 되면 이상한 값이 나온다.
                    '''
                    print_str += "target word : {} > :".format(dp.reverse_dictionary[neighbors[0]])

                    for k in range(1, similarity_number + 1, 1):
                        print_str += " {},".format(dp.reverse_dictionary[neighbors[k]])
                    print(print_str[:-1])

                if tSNE:
                    # T-SNE로 그려보기
                    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
                    low_dim_embs = tsne.fit_transform(final_embeddings[:tSNE_plot, :])
                    labels = [dp.reverse_dictionary[i] for i in range(tSNE_plot)]

                    figure = plt.figure(figsize=(18, 18))  # in inches
                    figure.suptitle("Visualizing Word2Vector using T-SNE")
                    for i, label in enumerate(labels):
                        x, y = low_dim_embs[i, :]
                        plt.scatter(x, y)
                        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right',
                                     va='bottom')
                    figure.savefig("Word2Vec_Using_TSNE.png", dpi=300)
                    plt.show()


if __name__ == "__main__":
    # optimizers_ selection = "Adam" or "RMSP" or "SGD"
    # weight_selection 은 encoder, decoder 임베디중 어떤것을 사용할 것인지 선택하는 변수
    # weight_sharing=True시 weight_selection="decoder"라고 설정해도 encoder의 embedding_matrix 로 강제 설정된다.
    Word2Vec(TEST=False, tSNE=True, model_name="Word2Vec", weight_selection="encoder",  # encoder or decoder
             vocabulary_size=50000, tSNE_plot=200, similarity_number=8,
             # similarity_number -> 비슷한 문자 출력 개수
             # num_skip : 하나의 문장당 num_skips 개의 데이터를 생성
             validation_number=30, embedding_size=128, batch_size=128, num_skips=2, window_size=1,
             negative_sampling=64, optimizer_selection="SGD", learning_rate=0.1, training_epochs=1000,
             display_step=1, weight_sharing=False)
else:
    print("word2vec imported")
