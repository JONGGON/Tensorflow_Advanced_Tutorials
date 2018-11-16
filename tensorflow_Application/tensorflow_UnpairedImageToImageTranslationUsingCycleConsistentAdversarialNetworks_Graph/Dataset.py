import glob
import os
import random
import urllib.request
import zipfile

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

'''
데이터셋은 아래에서 받았다.
https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip

나만의 이미지 데이터셋 만들기 - 텐서플로우의 API 만을 이용하여 만들자.
많은 것을 지원해주는 텐서플로우 API 만을 이용해서 만들 경우 코드가 굉장히 짧아지면서 빠르다.
하지만, 공부해야할 것이 꽤 많다. TFRecord, tf.data.Dataset API, tf.image API 등등 여러가지를 알아야 한다.

##################내가 생각하는 데이터 전처리기를 만드는 총 3가지 방법################# 
#총 3가지 방법이 있는 것 같다. 
1. numpy로 만들어서 feed_dict하는 방법 - feed_dict 자체가 파이런 런타임에서 텐서플로 런타임에서 데이터를 단일 스레드로 복사해서
이로인해 지연이 발생하고 속도가 느려진다.

2. tf.data.Dataset.from_tensor_slices 로 만든 다음에 그래프에 올려버리는 방법 - 아래의 첫번째 방법 - 현재 코드는 jpg, jpeg 데이터에만 가능
- 약간 어중간한 위치의 데이터 전처리 방법이다. 하지만, 1번보다는 빠르다

3. tf.data.TFRecordDataset를 사용하여 TRRecord(이진 파일, 직렬화된 입력 데이터)라는 텐서플로우 표준 파일형식으로 저장된 파일을 불러온 다음
  그래프에 올려버리는 방법 - 아래의 두번째 방법

<첫번째 방법>은 원래의 데이터파일을 불러와서 학습 한다.
<두번째 방법>은 TFRecord(텐서플로우의 표준 파일 형식)으로 원래의 데이터를 저장한뒤 불러와서 학습하는 방식이다.
<두번째 방법>이 빠르다.
<두번쨰 방법>은 모든데이터는 메모리의 하나의 블록에 저장되므로, 입력 파일이 개별로 저장된 <첫번째 방법>에 비헤
메모리에서 데이터를 읽는데 필요한 시간이 단축 된다.

구체적으로, 
<첫번째 방법>
1. 데이터를 다운로드한다. 데이터셋이 들어있는 파일명을 들고와 tf.data.Dataset.from_tensor_slices 로 읽어들인다.
2. tf.data.Dataset API 및 여러가지 유용한 텐서플로우 API 를 사용하여 학습이 가능한 데이터 형태로 만든다. 
    -> tf.read_file, tf.random_crop, tf.image.~ API를 사용하여 논문에서 설명한대로 이미지를 전처리하고 학습가능한 형태로 만든다.

<두번째 방법> - 메모리에서 데이터를 읽는 데 필요한 시간이 단축된다.
1. 데이터를 다운로드한다. 데이터가 대용량이므로 텐서플로의 기본 데이터 형식인 TFRecord(프로토콜버퍼, 직렬화) 형태로 바꾼다.
    -> 텐서플로로 바로 읽어 들일 수 있는 형식, 입력 파일들을 하나의 통합된 형식으로 변환하는 것(하나의 덩어리)
    -> 정리하자면 Tensor 인데, 하나의 덩어리 형태
2. tf.data.Dataset API 및 여러가지 유용한 텐서플로우 API 를 사용하여 학습이 가능한 데이터 형태로 만든다. 
    -> tf.read_file, tf.random_crop, tf.image.~ API를 사용하여 논문에서 설명한대로 이미지를 전처리하고 학습가능한 형태로 만든다.
'''


class Dataset(object):

    def __init__(self, DB_name="horse2zebra",
                 batch_size=1, TFRecord=False, use_TrainDataset=False, inference_size=(512, 512)):

        self.Dataset_Path = "Dataset"
        self.DB_name = DB_name
        self.inference_size = inference_size
        # 학습용 데이터인지 테스트용 데이터인지 알려주는 변수
        self.use_TrainDataset = use_TrainDataset

        # infernece_size의 최소 크기를 (256, 256)로 지정
        if not self.use_TrainDataset:
            if self.inference_size == None and self.inference_size[0] < 256 and self.inference_size[1] < 256:
                print("inference size는 (256,256)보다는 커야 합니다.")
                exit(0)
            else:
                self.height_size = inference_size[0]
                self.width_size = inference_size[1]

        # "{self.DB_name}.zip"의 파일의 크기는 미리 구해놓음. (미리 확인이 필요함.)
        if DB_name == "horse2zebra":
            self.file_size = 116871168
        else:
            print("Please enter ""DB_name"" correctly")
            print("The program is forcibly terminated.")
            exit(0)

        self.TFRecord = TFRecord
        if not os.path.exists(self.Dataset_Path):
            os.makedirs(self.Dataset_Path)

        self.url = "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/{}.zip".format(self.DB_name)
        self.dataset_folder = os.path.join(self.Dataset_Path, self.DB_name)
        self.dataset_zip = self.dataset_folder + ".zip"

        # 데이터셋 다운로드 한다.
        self.Preparing_Learning_Dataset()

        # TFRecord를 쓰기 위한 변수, use_TFRecord 가 True 이면 TFRecord 파일로 쓴다.
        if self.use_TrainDataset:
            self.batch_size = batch_size
            # (1). tf.data.Dataset.from_tensor_slices
            self.file_path_Alist = glob.glob(os.path.join(self.dataset_folder, "trainA/*"))
            self.file_path_Blist = glob.glob(os.path.join(self.dataset_folder, "trainB/*"))
            self.file_path_list = [self.file_path_Alist, self.file_path_Blist]

            # (2). TFRecord
            if self.TFRecord:

                self.TFRecord_train_path = os.path.join(self.dataset_folder, "TFRecord_train")
                if not os.path.exists(self.TFRecord_train_path):
                    os.makedirs(self.TFRecord_train_path)
                self.TFRecord_Apath = os.path.join(self.TFRecord_train_path, 'trainA.tfrecords')
                self.TFRecord_Bpath = os.path.join(self.TFRecord_train_path, 'trainB.tfrecords')
                self.TFRecord_path = [self.TFRecord_Apath, self.TFRecord_Bpath]
                # TFRecord 파일로 쓰기.
                self.TFRecordWriter()
        else:
            # Test Dataset은 무조건 하나씩 처리하자.
            self.batch_size = 1
            # (1). tf.data.Dataset.from_tensor_slices
            self.file_path_Alist = glob.glob(os.path.join(self.dataset_folder, "testA/*"))
            self.file_path_Blist = glob.glob(os.path.join(self.dataset_folder, "testB/*"))
            self.file_path_list = [self.file_path_Alist, self.file_path_Blist]

            # (2). TFRecord
            if self.TFRecord:

                self.TFRecord_val_path = os.path.join(self.dataset_folder, "TFRecord_test")
                if not os.path.exists(self.TFRecord_val_path):
                    os.makedirs(self.TFRecord_val_path)
                self.TFRecord_Apath = os.path.join(self.TFRecord_val_path,
                                                   'testA{}x{}.tfrecords'.format(self.height_size, self.width_size))
                self.TFRecord_Bpath = os.path.join(self.TFRecord_val_path,
                                                   'testB{}x{}.tfrecords'.format(self.height_size, self.width_size))
                self.TFRecord_path = [self.TFRecord_Apath, self.TFRecord_Bpath]
                # TFRecord 파일로 쓰기.
                self.TFRecordWriter()

    def __repr__(self):
        return "Dataset Loader"

    def iterator(self):

        if self.TFRecord:
            A_iterator, A_next_batch, A_length, B_iterator, B_next_batch, B_length = self.Using_TFRecordDataset()
        else:
            A_iterator, A_next_batch, A_length, B_iterator, B_next_batch, B_length = self.Using_TFBasicDataset()

        return A_iterator, A_next_batch, A_length, B_iterator, B_next_batch, B_length

    def Preparing_Learning_Dataset(self):

        # 1. 데이터셋 폴더가 존재하지 않으면 다운로드
        if not os.path.exists(self.dataset_folder):
            if not os.path.exists(self.dataset_zip):  # 데이터셋 압축 파일이 존재하지 않는 다면, 다운로드
                print("<<< {} Dataset Download required >>>".format(self.DB_name))
                urllib.request.urlretrieve(self.url, self.dataset_zip)
                print("<<< {} Dataset Download Completed >>>".format(self.DB_name))

            # "{self.DB_name}.tar.gz"의 파일의 크기는 미리 구해놓음. (미리 확인이 필요함.)
            elif os.path.exists(self.dataset_zip) and os.path.getsize(
                    self.dataset_zip) == self.file_size:  # 완전한 데이터셋 압축 파일이 존재한다면, 존재한다고 print를 띄워주자.
                print("<<< ALL {} Dataset Exists >>>".format(self.DB_name))

            else:  # 데이터셋 압축파일이 존재하긴 하는데, 제대로 다운로드 되지 않은 상태라면, 삭제하고 다시 다운로드
                print(
                    "<<< {} Dataset size must be : {}, but now size is {} >>>".format(self.DB_name, self.file_size,
                                                                                      os.path.getsize(
                                                                                          self.dataset_zip)))
                os.remove(self.dataset_zip)  # 완전하게 다운로드 되지 않은 기존의 데이터셋 압축 파일을 삭제
                print("<<< Deleting incomplete {} Dataset Completed >>>".format(self.DB_name))
                print("<<< we need to download {} Dataset again >>>".format(self.DB_name))
                urllib.request.urlretrieve(self.url, self.dataset_zip)
                print("<<< {} Dataset Download Completed >>>".format(self.DB_name))

            # 2. 완전한 압축파일이 다운로드 된 상태이므로 압축을 푼다
            with zipfile.ZipFile(self.dataset_zip, 'r') as zf:
                zf.extractall(path=self.Dataset_Path)

            print("<<< {} Unzip Completed >>>".format(os.path.basename(self.dataset_zip)))
            print("<<< {} Dataset now exists >>>".format(self.DB_name))
        else:
            print("<<< {} Dataset is already Exists >>>".format(self.DB_name))

    def _image_preprocessingOfBasic(self, image):

        # 이미지를 읽는다
        img = tf.read_file(image)
        img = tf.image.decode_image(img, channels=3)  # 채널을 꼭 지정해야한다. -> 안하니 오류가 발생한다.
        # tf.image.decode_image는 shape 정보를 반환하지 못하므로, 아래의 코드를 꼭 작성해야한다.
        img.set_shape([None, None, 3])
        img_cast = tf.cast(img, tf.float32)
        # gerator의 활성화 함수가 tanh이므로, 스케일을 맞춰준다.
        img_scaled = tf.subtract(tf.divide(img_cast, 127.5), 1.0)

        '''
        주의 
          BILINEAR = 0 -> 가장 가까운 화소값을 사용
          NEAREST_NEIGHBOR = 1 -> 인접한 4개 화소의 화소값과 거리비를 사용하여 결정
          BICUBIC = 2 -> 이것은 사용하지 않아야 한다. zero padding부분에도 값이 입혀진다(인접한 16개 화소의 화소밗과 거리에 따른 가중치의 곱을 사용하기때문)
          AREA = 3 -> 사이즈 줄일때 사용
        '''
        # Train Dataset 에서만 동작하게 하기 위함
        if self.use_TrainDataset:
            # 이미지를 키운다
            expanded_area = tf.random_uniform((1,), 0, 31, dtype=tf.int32)[0]  # 0 ~ 30 의 값으로 키워서 자른다.(랜덤)
            left = tf.random_uniform((1,), 0, 1, dtype=tf.float32)[0]
            right = tf.random_uniform((1,), 0, 1, dtype=tf.float32)[0]

            input_resized = tf.cond(tf.less(left, right), lambda: tf.image.resize_images(images=img_scaled, size=(
                tf.shape(img_scaled)[0] + expanded_area, tf.shape(img_scaled)[1] + expanded_area), method=1),
                                    lambda: img_scaled)

            # 이미지를 원본 크기로 자른다.
            input_cropped = tf.random_crop(input_resized, size=(
                tf.shape(img_scaled)[0], tf.shape(img_scaled)[1], tf.shape(img_scaled)[-1]))
            input = input_cropped
        else:
            # 이미지 사이즈를 self.height_size x self.width_size 으로 조정한다.
            input = tf.image.resize_images(img_scaled, size=(self.height_size, self.width_size), method=2)

        return input

    def _image_preprocessingOfTFRecord(self, image):

        # 1. 이미지를 읽는다.
        feature = {'image': tf.FixedLenFeature([], tf.string),
                   'height': tf.FixedLenFeature([], tf.int64),
                   'width': tf.FixedLenFeature([], tf.int64),
                   'depth': tf.FixedLenFeature([], tf.int64)}
        parser = tf.parse_single_example(image, features=feature)
        img_decoded_raw = tf.decode_raw(parser['image'], tf.float32)
        height = tf.cast(parser['height'], tf.int32)
        width = tf.cast(parser['width'], tf.int32)
        depth = tf.cast(parser['depth'], tf.int32)
        img_decoded = tf.reshape(img_decoded_raw, (height, width, depth))
        # gerator의 활성화 함수가 tanh이므로, 스케일을 맞춰준다.
        img_scaled = tf.subtract(tf.divide(img_decoded, 127.5), 1.0)

        '''
        주의 
          BILINEAR = 0 -> 가장 가까운 화소값을 사용
          NEAREST_NEIGHBOR = 1 -> 인접한 4개 화소의 화소값과 거리비를 사용하여 결정
          BICUBIC = 2 -> 이것은 사용하지 않아야 한다. zero padding부분에도 값이 입혀진다(인접한 16개 화소의 화소밗과 거리에 따른 가중치의 곱을 사용하기때문)
          AREA = 3 -> 사이즈 줄일때 사용
        '''
        # Train Dataset 에서만 동작하게 하기 위함
        if self.use_TrainDataset:
            # 이미지를 키운다
            expanded_area = tf.random_uniform((1,), 0, 31, dtype=tf.int32)[0]  # 0 ~ 30 의 값으로 키워서 자른다.(랜덤)
            left = tf.random_uniform((1,), 0, 1, dtype=tf.float32)[0]
            right = tf.random_uniform((1,), 0, 1, dtype=tf.float32)[0]

            input_resized = tf.cond(tf.less(left, right), lambda: tf.image.resize_images(images=img_scaled, size=(
                tf.shape(img_scaled)[0] + expanded_area, tf.shape(img_scaled)[1] + expanded_area), method=1),
                                    lambda: img_scaled)

            # 이미지를 원본 크기로 자른다.
            input_cropped = tf.random_crop(input_resized, size=(
                tf.shape(img_scaled)[0], tf.shape(img_scaled)[1], tf.shape(img_scaled)[-1]))
            input = input_cropped
        else:
            # 이미지 사이즈를 self.height_size x self.width_size 으로 조정한다.
            input = tf.image.resize_images(img_scaled, size=(self.height_size, self.width_size), method=2)

        return input

    # TFRecord를 만들기 위해 이미지를 불러올때 쓴다.
    def load_image(self, address):

        img = cv2.imread(address)

        # TEST = True 일 때
        if not self.use_TrainDataset:
            img = cv2.resize(img, (self.width_size, self.height_size), interpolation=cv2.INTER_CUBIC)

        # RGB로 바꾸기
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        return img

    def TFRecordWriter(self):

        # http: // machinelearninguru.com / deep_learning / data_preparation / tfrecord / tfrecord.html 참고했다.
        # TFRecord로 바꾸기
        print("<<< Using TFRecord format >>>")

        for TFRecord_path, file_path_list in zip(self.TFRecord_path, self.file_path_list):

            if not os.path.isfile(TFRecord_path):  # TFRecord 파일이 존재하지 않은 경우
                print("<<< Making {} >>>".format(os.path.basename(TFRecord_path)))
                with tf.python_io.TFRecordWriter(TFRecord_path) as writer:  # TFRecord로 쓰자
                    if self.use_TrainDataset:
                        random.shuffle(file_path_list)
                    for image_address in tqdm(file_path_list):
                        img = self.load_image(image_address)
                        '''넘파이 배열의 값을 바이트 스트링으로 변환한다.
                        tf.train.BytesList, tf.train.Int64List, tf.train.FloatList 을 지원한다.
                        '''
                        feature = \
                            {
                                'image': tf.train.Feature(
                                    bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(img.tostring())])),
                                'height': tf.train.Feature(
                                    int64_list=tf.train.Int64List(value=[img.shape[0]])),
                                'width': tf.train.Feature(
                                    int64_list=tf.train.Int64List(value=[img.shape[1]])),
                                'depth': tf.train.Feature(
                                    int64_list=tf.train.Int64List(value=[img.shape[-1]])),
                            }
                        example = tf.train.Example(features=tf.train.Features(feature=feature))
                        # 파일로 쓰자.
                        writer.write(example.SerializeToString())
                print("<<< Making {} is Completed >>>".format(os.path.basename(TFRecord_path)))
            else:  # TFRecord가 존재할 경우
                print("<<< {} already exists >>>".format(os.path.basename(TFRecord_path)))

    # tf.data.TFRecordDataset를 사용하는 방법 - TRRecord(이진 파일, 직렬화된 입력 데이터)라는 텐서플로우 표준 파일형식으로 저장된 파일을 불러와서 처리하기
    def Using_TFRecordDataset(self):

        A_length = sum(1 for _ in tf.python_io.tf_record_iterator(self.TFRecord_Apath))
        B_length = sum(1 for _ in tf.python_io.tf_record_iterator(self.TFRecord_Bpath))

        # 4. TFRecordDataset()사용해서 읽어오기
        A_dataset = tf.data.TFRecordDataset(self.TFRecord_Apath)
        B_dataset = tf.data.TFRecordDataset(self.TFRecord_Bpath)
        A_dataset = A_dataset.map(self._image_preprocessingOfTFRecord, num_parallel_calls=10)
        B_dataset = B_dataset.map(self._image_preprocessingOfTFRecord, num_parallel_calls=10)

        if self.use_TrainDataset:
            A_dataset = A_dataset.shuffle(buffer_size=1000).repeat().batch(self.batch_size).prefetch(self.batch_size)
            B_dataset = B_dataset.shuffle(buffer_size=1000).repeat().batch(self.batch_size).prefetch(self.batch_size)
        else:
            A_dataset = A_dataset.repeat().batch(self.batch_size).prefetch(self.batch_size)
            B_dataset = B_dataset.repeat().batch(self.batch_size).prefetch(self.batch_size)

        # Using_TFBasicDataset와 형태를 맞추기 위함이다. -> 사실 여기선 dataset.make_one_shot_iterator()을 사용해도 된다.
        A_iterator = A_dataset.make_initializable_iterator()
        B_iterator = B_dataset.make_initializable_iterator()
        # tf.python_io.tf_record_iterator는 무엇인가 ? TFRecord 파일에서 레코드를 읽을 수 있는 iterator이다.

        return A_iterator, A_iterator.get_next(), A_length, B_iterator, B_iterator.get_next(), B_length

    # tf.data.Dataset.from_tensor_slices 을 사용하는 방법 - 파일명 리스트에서 이미지를 불러와서 처리하기
    def Using_TFBasicDataset(self):

        A_length = len(self.file_path_Alist)
        B_length = len(self.file_path_Blist)

        if self.use_TrainDataset:
            random_file_path_Alist_Tensor = tf.random_shuffle(tf.constant(self.file_path_Alist))  # tensor에 데이터셋 리스트를 담기
            random_file_path_Blist_Tensor = tf.random_shuffle(tf.constant(self.file_path_Blist))  # tensor에 데이터셋 리스트를 담기
            A_dataset = tf.data.Dataset.from_tensor_slices(random_file_path_Alist_Tensor)
            B_dataset = tf.data.Dataset.from_tensor_slices(random_file_path_Blist_Tensor)
        else:
            A_dataset = tf.data.Dataset.from_tensor_slices(tf.constant(self.file_path_Alist))
            B_dataset = tf.data.Dataset.from_tensor_slices(tf.constant(self.file_path_Blist))

        A_dataset = A_dataset.map(self._image_preprocessingOfBasic, num_parallel_calls=10)
        B_dataset = B_dataset.map(self._image_preprocessingOfBasic, num_parallel_calls=10)
        '''
        buffer_size: A `tf.int64` scalar `tf.Tensor`, representing the
        number of elements from this dataset from which the new
        dataset will sample.

        dataset.buffer_size란 정확히 무엇인가? shuffling 할 때 몇개를 미리 뽑아서 랜덤하게 바꾸는건데 이상적으로 봤을 때 buffer_size가
        파일 개수 만큼이면 좋겠지만, 이미지 자체의 순서를 바꾸는 것이기 때문에 메모리를 상당히 많이 먹는다. (메모리가 16기가인 컴퓨터에서
        buffer_size = 5000 만되도 컴퓨터가 강제 종료 된다.)
        따라서 dataset.shuffle 의 매개 변수인 buffer_size를  1000정도로 로 설정(buffer_size가 1이라는 의미는 사용 안한다는 의미)

        더 좋은 방법? -> 파일명이 들어있는 리스트가  tf.data.Dataset.from_tensor_slices의 인자로 들어가기 전에 미리 섞는다.(tf.random_shuffle 을 사용)
        문제점1 -> tf.random_shuffle을 사용하려면 dataset.make_one_shot_iterator()을 사용하면 안된다. tf.random_shuffle을 사용하지 않고
        파일리스트를 램던함게 섞으려면 random 모듈의 shuffle을 사용해서 미리 self.file_path_list을 섞는다.
        문제점2 -> 한번 섞고 말아버린다. -> buffer_size를 자기 컴퓨터의 메모리에 맞게 최대한으로 써보자.
        '''
        # dataset = dataset.shuffle(buffer_size=1).repeat().batch(self.batch_size)
        if self.use_TrainDataset:
            A_dataset = A_dataset.shuffle(buffer_size=1000).repeat().batch(self.batch_size).prefetch(self.batch_size)
            B_dataset = B_dataset.shuffle(buffer_size=1000).repeat().batch(self.batch_size).prefetch(self.batch_size)
        else:
            A_dataset = A_dataset.repeat().batch(self.batch_size).prefetch(self.batch_size)
            B_dataset = B_dataset.repeat().batch(self.batch_size).prefetch(self.batch_size)

        '''
        위에서 tf.random_shuffle을 쓰고 아래의 make_one_shot_iterator()을 쓰면 오류가 발생한다. - stateful 관련 오류가 뜨는데, 추 후 해결 되겠지...
        이유가 궁금하다면 아래의 웹사이트를 참고하자.
        https://stackoverflow.com/questions/44374083/tensorflow-cannot-capture-a-stateful-node-by-value-in-tf-contrib-data-api
        '''
        # iterator = dataset.make_one_shot_iterator()

        # tf.random_shuffle을 쓰려면 아래와 같이 make_initializable_iterator을 써야한다. - stack overflow 에서 찾음
        A_iterator = A_dataset.make_initializable_iterator()
        B_iterator = B_dataset.make_initializable_iterator()

        return A_iterator, A_iterator.get_next(), A_length, B_iterator, B_iterator.get_next(), B_length


''' 
to reduce model oscillation [14], we follow
Shrivastava et al’s strategy [45] and update the discriminators
using a history of generated images rather than the ones
produced by the latest generative networks. We keep an image
buffer that stores the 50 previously generated images.

imagePool 클래스
# https://github.com/xhujoy/CycleGAN-tensorflow/blob/master/utils.py 를 참고해서 변형했다.
'''


class ImagePool(object):

    def __init__(self, image_pool_size=50):

        self.image_pool_size = image_pool_size
        self.image_count = 0
        self.images_appender = []

    def __repr__(self):
        return "Image Pool class"

    def __call__(self, images=None):

        # 1. self.image_pool_size 사이즈가 0이거나 작으면, ImagePool을 사용하지 않는다.
        if self.image_pool_size <= 0:
            return images

        '''2. self.num_img 이 self.image_pool_size 보다 작으면, self.image_count을 하나씩 늘려주면서
        self.images_appender에 images를 추가해준다.
        self.image_pool_size 개 self.images_appender에 이전 images를 저장한다.'''
        if self.image_count < self.image_pool_size:
            self.images_appender.append(images)
            self.image_count += 1
            return images

        # copy에 대한 내용은 본 프로젝트의 copy_example.py를 참고!!!
        # np.random.rand()는 0~1 사이의 무작위 값을 출력한다.
        if np.random.rand() > 0.5:
            index = np.random.randint(low=0, high=self.image_pool_size, size=None)
            past_image1 = self.images_appender[index][0]
            self.images_appender[index][0] = images[0]
            index = np.random.randint(low=0, high=self.image_pool_size, size=None)
            past_image2 = self.images_appender[index][1]
            self.images_appender[index][1] = images[1]
            return past_image1, past_image2
        else:
            return images


if __name__ == "__main__":
    '''
    Dataset "horse2zebra" 만..
    '''
    dataset = Dataset(DB_name="horse2zebra", batch_size=1, use_TFRecord=True, use_TrainDataset=True)
    A_iterator, A_next_batch, A_length, B_iterator, B_next_batch, B_length = dataset.iterator()

else:
    print("Dataset imported")
