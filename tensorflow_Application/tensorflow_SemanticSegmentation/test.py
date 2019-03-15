import timeit

from core.dataset import *
from core.function_list import *
from core.html import *
from core.make_numpy_weight import *


def db_test(model_name="",
            Inputsize_limit=(256, 256),
            weights_to_numpy=True,
            using_latest_weight=True,
            WeightSelection=250,
            Dropout_rate=1,
            masking=True,
            masking_Image_save_path="masking_image",
            HTML_Report=True,
            HTML_Font_Size=30):
    tf.reset_default_graph()
    meta_path = glob.glob(os.path.join(model_name, '*.meta'))
    if meta_path:
        print("<<< Graph가 존재 합니다. >>>")
    else:
        print("<<< Graph가 존재 하지 않습니다. 그래프를 그려 주세요. - only_draw_graph = True >>>")
        print("<<< 강제 종료 합니다. >>>")
        exit(0)

    if os.path.exists(os.path.join(model_name, "information.npy")):
        class_number, input_range, Accesskey = np.load(
            os.path.join(model_name, "information.npy"))
        class_number = int(class_number)
        print("<<< training DB의 class number : {} 입니다. >>>".format(class_number))
    else:
        print("<<< {} 폴더 안에 {} 가 존재하지 않습니다. >>>".format(model_name, "information.npy"))
        print("<<< test 불가 >>>")
        exit(0)

    # print(tf.get_default_graph()) #기본그래프이다.
    graph = tf.Graph()  # 내 그래프로 설정한다.- 혹시라도 나중에 여러 그래프를 사용할 경우를 대비
    with graph.as_default():  # as_default()는 JG를 기본그래프로 설정한다.

        saver = tf.train.import_meta_graph(meta_path[0], clear_devices=True)  # meta graph 읽어오기
        if saver == None:
            print("<<< meta 파일을 읽을 수 없습니다. >>>")
            print("<<< 강제 종료합니다. >>>")
            exit(0)

        '''
        WHY? 아래 N줄의 코드를 적어 주지 않으면 오류가 발생한다. 
        -> 단순히 그래프를 가져오고 가중치를 복원하는 것만으로는 안된다. 세션을 실행할때 인수로 사용할 변수에 대한 
        추가 접근을 제공하지 않기 때문에 아래와 같이 get_colltection으로 입,출력 변수들을 불러와서 다시 사용 해야 한다.
        '''
        # Test Dataset에서 가져오기
        dataset = Dataset(batch_size=1, use_TrainDataset=False, input_range=input_range)
        next_batch, data_length = dataset.iterator()

        collection_list = tf.get_collection(Accesskey)
        x = collection_list[0]
        target = collection_list[1]
        G = collection_list[2]
        dis_loss = collection_list[3]
        keep_probability = collection_list[4]
        conv_list = collection_list[5:]

        t = tf.argmax(target, -1)
        p = tf.argmax(G, -1)

        # mean Intersection Over Union 계산
        mIOU, update_op_mIOU = tf.metrics.mean_iou(labels=t, predictions=p,
                                                   num_classes=class_number)

        # pixel accuracy 계산
        true = tf.ones_like(t)
        false = tf.zeros_like(t)
        temp = tf.where(tf.equal(t, p), x=true, y=false)
        numerator = tf.count_nonzero(temp, dtype=tf.int32)
        denominator = tf.size(temp, out_type=tf.int32)
        pacc = tf.divide(numerator, denominator)

        with tf.Session(graph=graph) as sess:
            sess.run(tf.local_variables_initializer())  # mion , pixel accuracy를 위해서 초기화 해줘야 한다.
            ckpt = tf.train.get_checkpoint_state(model_name)
            if ckpt == None:
                print("<<< checkpoint file does not exist>>>")
                print("<<< Exit the program >>>")
                exit(0)

            if (ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path)):
                print("<<< all variable retored except for optimizer parameter >>>")
                if using_latest_weight:
                    model_path = ckpt.model_checkpoint_path
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    Latest_WeightSelection = os.path.basename(ckpt.model_checkpoint_path).split("-")[-1]
                    print("<<< Restore {} checkpoint!!! >>>".format(Latest_WeightSelection))
                else:
                    weight_list = ckpt.all_model_checkpoint_paths
                    Condition = False
                    for i, wl in enumerate(weight_list):
                        if int(os.path.basename(wl).split("-")[-1]) == WeightSelection:
                            model_path = wl
                            saver.restore(sess, wl)
                            print("<<< Restore {} checkpoint!!! >>>".format(WeightSelection))
                            Condition = True
                            break
                        else:
                            Condition = False

                    if Condition == False:
                        print("<<< 선택한 {} 가중치가 없습니다. >>>".format(WeightSelection))
                        print("<<< 강제 종료합니다. >>>")
                        exit(0)

			# numpy 로 학습 완료된 가중치 저장하기
            if weights_to_numpy:
                # weight가 저장될 장소
                folder_path = model_name + "_LearnedInfo"

                if using_latest_weight:
                    make_numpy_weight(model_path, folder_path, input_range,
                                      WeightSelection=Latest_WeightSelection)
                else:
                    make_numpy_weight(model_path, folder_path, input_range,
                                      WeightSelection=WeightSelection)

            total_time = 0
            result_txt = "evaluation.txt"
            if os.path.exists(result_txt):
                print("<<< {} 삭제 >>>\n".format(result_txt))
                os.remove(result_txt)

            for i in range(1, data_length + 1, 1):
                print("<-- testDataset Number : {} / {} -->".format(i, data_length))
                x_numpy, target_numpy, file_name = sess.run(next_batch)
                file_name = file_name[0].decode("utf-8")

                # 입력 이미지가 Inputsize_limit[0] x Inputsize_limit[1] 이하이면, exit()
                if x_numpy.shape[1] < Inputsize_limit[0] or x_numpy.shape[2] < Inputsize_limit[1]:
                    print("<<< 입력된 이미지 크기는 {} x {} 입니다. >>>".format(x_numpy.shape[1], x_numpy.shape[2]))
                    print("<<< 입력되는 이미지 크기는 {} x {} 보다 크거나 같아야 합니다. >>>".format(Inputsize_limit[0],
                                                                                Inputsize_limit[1]))
                    print("<<< 강제 종료 합니다. >>>")
                    exit(0)

                start_time = timeit.default_timer()

                result = sess.run([G, dis_loss] + conv_list + [update_op_mIOU, pacc],
                                  feed_dict={x: x_numpy, target: target_numpy, keep_probability: Dropout_rate})

                # 평가 척도
                # pixel accuracy, mean intersection over union
                pixelacc = result[-1]
                meanIOU = sess.run(mIOU)

                # 결과 - segmentation map
                input = x_numpy[0]
                target_segmented_image = np.argmax(target_numpy[0], axis=-1)
                pred_segmented_image = np.argmax(result[0][0], axis=-1)

                loss = result[1]
                # conv_layer = result[2:]
                end_time = timeit.default_timer()
                running_time = end_time - start_time
                total_time += running_time

                print("<<< {}'s file name : {} >>>".format(i, file_name))
                print("<<< loss : {} >>>".format(loss))
                print("<<< pixel accuracy : {:0.3f}%".format(pixelacc * 100))
                print("<<< meanIOU : {:0.3f}%".format(meanIOU * 100))
                print("<<< 실행 시간 : {:0.3f}ms >>>".format((running_time) * 1000))

                with open(result_txt, "a+") as f:
                    f.write("{}'s file name : {} / Loss -> {}\n".format(i, file_name, loss))
                    f.write("Loss -> {}\n".format(loss))
                    f.write("pixel accuracy -> {:0.3f}%\n".format(pixelacc))
                    f.write("meanIOU -> {:0.3f}%\n".format(meanIOU))
                    f.write("실행 시간 : {:0.3f}ms\n\n".format(running_time * 1000))

                if masking:
                    # file_name=str(i+1)+"_"+file_name ? -> html 그릴때 str(i+1)기준으로 구분하기 위함.
                    draw_segmentation(masking_Image_save_path=masking_Image_save_path,
                                      file_name=str(i) + "_" + file_name, input=input,
                                      target=target_segmented_image, prediction=pred_segmented_image,
                                      class_number=class_number,
                                      input_range=input_range)
                print("\n")
            print("<<< 평균 실행 시간 : {:0.3f}ms >>>".format((total_time / data_length) * 1000))

            # HTML로 쓰기
            generated_image_list = glob.glob(masking_Image_save_path + "/*")
            if generated_image_list and HTML_Report:
                HTML(path=masking_Image_save_path, character_size=HTML_Font_Size)
                print("<<< Making HTML Report is Completed!!! >>>")
            else:
                print("<<< HTML_Report = False or generated_image_list=[] 입니다. >>>")
