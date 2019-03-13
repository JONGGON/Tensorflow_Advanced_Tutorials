import glob
import os
import random

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm


# Dataloader
class Dataset(object):

    def __init__(self, batch_size=1, use_TrainDataset=False, input_range="0~1",
                 Augmentation_algorithm=0):

        # self.origin_train_Dataset_Path = "trainDataset"에 있는 데이터를 지우면 안된다.
        # self.origin_test_Dataset_Path = "testDataset"에 있는 데이터를 지우면 안된다.

        self.Dataset_Path = "Dataset"
        self.input_range = input_range

        # 학습용 데이터인지 테스트용 데이터인지 알려주는 변수
        self.use_TrainDataset = use_TrainDataset
        self.Augmentation_algorithm = Augmentation_algorithm

        if self.use_TrainDataset:
            # 원본 트레이닝 데이터가 저장된 폴더
            self.batch_size = batch_size
            self.origin_train_Dataset_Path = "trainDataset"
            self.TFRecord_path = []

            if not os.path.exists(self.origin_train_Dataset_Path):
                print("<<< {} 폴더가 없습니다. >>>".format(self.origin_train_Dataset_Path))
                os.makedirs(self.origin_train_Dataset_Path)
                print("<<< {} 폴더 만들고 강제 종료 합니다. (~.bmp , ~.npy) 파일들을 넣어주세요 >>>".format(
                    self.origin_train_Dataset_Path))
                exit(0)

            self.image_list = glob.glob("{}/*.bmp".format(self.origin_train_Dataset_Path))
            self.label_list = glob.glob("{}/*.npy".format(self.origin_train_Dataset_Path))

            if self.image_list == [] or self.label_list == []:
                # 데이터가 들어있는지 확인하는 용도
                print("<<< {} 폴더안에 원본 ~.bmp train dataset 또는 원본 ~.npy train dataset이 없습니다. >>>".format(
                    self.origin_train_Dataset_Path))
                print("강제 종료 합니다.")  # 데이터가 추가될 때를 대비
                exit(0)

            else:  # TFRecord로 쓰기
                if not os.path.exists(self.Dataset_Path):
                    os.makedirs(self.Dataset_Path)
                self.TFRecord_path = os.path.join(self.Dataset_Path,
                                                  'train{}.tfrecords'.format(len(
                                                      self.image_list)))  # self.origin_train_Dataset_Path 에 있는 데이터 길이로 TFRecord를 만들지 말지를 판단한다.
                self.TFRecordWriter()

        else:
            # Test=True 일 때 무조건 하나씩 처리하기 - 내가 만든 visualize 함수가 batch_size = 1에 대해서만 동작함
            self.batch_size = 1
            # 원본 테스트 데이터가 저장된 폴더
            self.origin_test_Dataset_Path = "testDataset"

            if not os.path.exists(self.origin_test_Dataset_Path):
                print("<<< {} 폴더가 없습니다. >>>".format(self.origin_test_Dataset_Path))
                os.makedirs(self.origin_test_Dataset_Path)
                print("<<< {} 폴더 만들고 강제 종료 합니다. (~.bmp , ~.npy) 파일들을 넣어주세요 >>>".format(
                    self.origin_test_Dataset_Path))
                exit(0)

            self.image_list = glob.glob("{}/*.bmp".format(self.origin_test_Dataset_Path))
            self.label_list = glob.glob("{}/*.npy".format(self.origin_test_Dataset_Path))

            if self.image_list == [] or self.label_list == []:

                # 데이터가 들어있는지 확인하는 용도
                print("<<< {} 폴더안에 원본 BoardImage.bmp test dataset 또는 원본 Class.npy test dataset이 없습니다. >>>".format(
                    self.origin_test_Dataset_Path))
                print("강제 종료 합니다.")  # 데이터가 추가될 때를 대비
                exit(0)

            else:  # TFRecord로 쓰기
                if not os.path.exists(self.Dataset_Path):
                    os.makedirs(self.Dataset_Path)
                self.TFRecord_path = os.path.join(self.Dataset_Path,
                                                  'test{}.tfrecords'.format(len(
                                                      self.image_list)))  # self.origin_train_Dataset_Path 에 있는 데이터 길이로 TFRecord를 만들지 말지를 판단한다.
                self.TFRecordWriter()

    def __repr__(self):
        return "Dataset Loader"

    def load_data(self, path):

        # 1. image
        path_basename = os.path.basename(path)
        # 제외될 위치 - [Conf]가 구분자이다.
        path_except_position = path_basename.find("[Conf]")
        for label in self.label_list:
            # 2. label
            label_basename = os.path.basename(label)

            # 제외될 위치 - [Conf]가 구분자이다.
            label_except_position = label_basename.find("[Conf]")
            if path_basename[:path_except_position] == label_basename[:label_except_position]:
                image = cv2.imread(path)  # BGR로 읽는다.
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB로 바꾼다
                label = np.load(label).astype(np.uint8)
                # cv2.imshow("image", cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), (256,256)))
                # cv2.imshow("1", cv2.resize(label[:,:,0], (256,256)))
                # cv2.imshow("2", cv2.resize(label[:,:,1], (256,256)))
                # cv2.waitKey(0)
                return image, label

    def key_func(self, path):

        image = cv2.imread(path)
        # RMS 구하기
        rms_error = np.sqrt(np.mean(np.square(image)))
        return rms_error

    def TFRecordWriter(self):

        print("<<< Using TFRecord format >>>")
        if not os.path.isfile(self.TFRecord_path) or os.path.getsize(
                self.TFRecord_path) == 0:  # TFRecord 파일이 존재하지 않은 경우 , 잘못 만들어진 경우:
            print("<<< Making {} >>>".format(os.path.basename(self.TFRecord_path)))
            if self.use_TrainDataset:
                random.shuffle(self.image_list)  # 한번 섞자
            else:  # 입력 이미지의 RMS 로 정렬하기
                self.image_list = sorted(self.image_list, key=lambda path: self.key_func(path))
            with tf.python_io.TFRecordWriter(self.TFRecord_path) as writer:  # TFRecord로 쓰자
                for path in tqdm(self.image_list):
                    image, label = self.load_data(path)

                    file_name = os.path.basename(path).split('[Conf]')[0]
                    '''넘파이 배열의 값을 바이트 스트링으로 변환한다.
                    tf.train.BytesList, tf.train.Int64List, tf.train.FloatList 을 지원한다.
                    '''
                    feature = \
                        {
                            'image': tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(image.tostring())])),
                            'label': tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(label.tostring())])),

                            'image_height': tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[image.shape[0]])),
                            'image_width': tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[image.shape[1]])),
                            'image_depth': tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[image.shape[-1]])),
                            'label_depth': tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[label.shape[-1]])),

                            'file_name': tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(file_name, encoding="utf-8")])),
                        }

                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    # 파일로 쓰자.
                    writer.write(example.SerializeToString())
            print("<<< Making {} is Completed >>>".format(os.path.basename(self.TFRecord_path)))
        else:  # TFRecord가 존재할 경우
            print("<<< {} already exists >>>".format(os.path.basename(self.TFRecord_path)))

    '''이 부분에서 TFRecordWriter함수안에 있는 load_image함수의 기능을 넣어도 되지만, 
       여기서 이미지를 자르고 정규화 하는 작업을 하면 그래프에 연산을 추가하게 되고 연산량이 많아지므로 하지 않는다.
       단지 읽어오는 역할만 한다.'''

    def _image_preprocessing(self, data):

        # 1. 이미지를 읽는다.
        feature = {'image': tf.FixedLenFeature([], tf.string),
                   'label': tf.FixedLenFeature([], tf.string),
                   'image_height': tf.FixedLenFeature([], tf.int64),
                   'image_width': tf.FixedLenFeature([], tf.int64),
                   'image_depth': tf.FixedLenFeature([], tf.int64),
                   'label_depth': tf.FixedLenFeature([], tf.int64),
                   'file_name': tf.FixedLenFeature([], tf.string)}
        # <<< here >>>
        '''
            "FixedLenFeature", ["shape", "dtype", "default_value"])):
          """Configuration for parsing a fixed-length input feature.
        
          To treat sparse input as dense, provide a `default_value`; otherwise,
          the parse functions will fail on any examples missing this feature.
        
          Fields:
            shape: Shape of input data.
            dtype: Data type of input.
            default_value: Value to be used if an example is missing this feature. It
                must be compatible with `dtype` and of the specified `shape`.
        '''
        parser = tf.parse_single_example(data, features=feature)
        image = tf.decode_raw(parser['image'], tf.uint8)  # uint8로 보냈으니, uint8로 받는게 당연 - 자료형 조심
        label = tf.decode_raw(parser['label'], tf.uint8)  # uint8로 보냈으니, uint8로 받는게 당연 - 자료형 조심
        image_height = tf.cast(parser['image_height'], tf.int32)
        image_width = tf.cast(parser['image_width'], tf.int32)
        image_depth = tf.cast(parser['image_depth'], tf.int32)
        label_depth = tf.cast(parser['label_depth'], tf.int32)
        file_name = parser['file_name']  # string 형은 무조건 [] 가 씌워져서 나온다.

        image = tf.reshape(image, (image_height, image_width, image_depth))
        label = tf.reshape(label, (image_height, image_width, label_depth))

        # float32로 바꿔줘야 한다.
        image = tf.cast(image, tf.float32)

        label = tf.cast(label, tf.float32)
        label = tf.divide(label, 255.0)

        if self.batch_size > 1:
            print("\n batch size > 1 인 경우에는 Augmentation Algorithm 1 -> random padding algorithm 은 동작 안함")

        if self.use_TrainDataset and self.Augmentation_algorithm >= 1 and self.batch_size == 1:
            print("\nAugmentation Algorithm 1 -> random padding algorithm")
            random_height = tf.random_uniform(shape=(1,), minval=0, maxval=7, dtype=tf.int32)[0]
            random_width = tf.random_uniform(shape=(1,), minval=0, maxval=7, dtype=tf.int32)[0]
            paddings_height = tf.zeros(shape=(1, 2), dtype=tf.int32) + random_height
            paddings_width = tf.zeros(shape=(1, 2), dtype=tf.int32) + random_width
            paddings_depth = tf.zeros(shape=(1, 2), dtype=tf.int32)

            paddings = tf.concat([paddings_height, paddings_width, paddings_depth], axis=0)
            standard = tf.random_uniform((1,), 0, 1, dtype=tf.float32)[0]

            image, label = tf.cond(tf.greater_equal(standard, 0.5), \
                                   lambda: [
                                       tf.pad(image, paddings, mode="CONSTANT",
                                              constant_values=0),
                                       tf.pad(label, paddings, mode="CONSTANT", constant_values=0)], \
                                   lambda: [image, label])

        if self.input_range == "0~1":
            image = tf.divide(image, 255.0)
        elif self.input_range == "-1~1":
            image = tf.subtract(tf.divide(image, 127.5), 1.0)

        return image, label, file_name

    def _data_augmentation(self, image, label, file_name):

        if self.Augmentation_algorithm >= 2:
            print("Augmentation Algorithm 2 -> run rotation1 algorithm")

            standard = tf.random_uniform((1,), 0, 1, dtype=tf.float32)[0]

            IntChoice = tf.random_uniform((1,), 0, 3, dtype=tf.int32)[0]

            image, label = tf.cond(tf.logical_and(tf.greater_equal(standard, 0.5), tf.equal(IntChoice, 0)), \
                                   lambda: [
                                       tf.image.rot90(image, k=1),
                                       tf.image.rot90(label, k=1)],
                                   lambda: [image, label])

            image, label = tf.cond(tf.logical_and(tf.greater_equal(standard, 0.5), tf.equal(IntChoice, 1)), \
                                   lambda: [
                                       tf.image.rot90(image, k=2),
                                       tf.image.rot90(label, k=2)], \
                                   lambda: [image, label])

            image, label = tf.cond(tf.logical_and(tf.greater_equal(standard, 0.5), tf.equal(IntChoice, 2)), \
                                   lambda: [
                                       tf.image.rot90(image, k=3),
                                       tf.image.rot90(label, k=3)], \
                                   lambda: [image, label])

            if self.Augmentation_algorithm >= 3:
                print("Augmentation Algorithm 3 -> run rotation2 algorithm")
                standard = tf.random_uniform((1,), 0, 1, dtype=tf.float32)[0]

                rotation = tf.random_uniform((1,), -10, 11, dtype=tf.int32)[0]
                rotation = tf.cast(rotation, dtype=tf.float32)

                image, label = tf.cond(tf.greater_equal(standard, 0.5), \
                                       lambda: [
                                           tf.contrib.image.rotate(image, rotation * np.pi / 180,
                                                                   interpolation='NEAREST'),
                                           tf.contrib.image.rotate(label, rotation * np.pi / 180,
                                                                   interpolation='NEAREST')], \
                                       lambda: [image, label])

            if self.Augmentation_algorithm >= 4:
                print("Augmentation Algorithm 4 -> run translation algorithm")
                standard = tf.random_uniform((1,), 0, 1, dtype=tf.float32)[0]

                translation = tf.random_uniform((2,), -10, 11, dtype=tf.int32)  # 상하로 조금씩 진동하는 효과 - 주의 dtype이 int면 안된다.
                translation = tf.cast(translation, dtype=tf.float32)

                image, label = tf.cond(tf.greater_equal(standard, 0.5), \
                                       lambda: [
                                           tf.contrib.image.translate(image, translation, interpolation="NEAREST"),
                                           tf.contrib.image.translate(label, translation, interpolation="NEAREST")], \
                                       lambda: [image, label])

            # self.Augmentation_algorithm <= 5 일 경우 random crop - 위 알고리즘들 포함
            '''
              BILINEAR = 0
              NEAREST_NEIGHBOR = 1
              BICUBIC = 2
              AREA = 3
            '''
            if self.Augmentation_algorithm >= 5:
                print("Augmentation Algorithm 5 -> run random crop algorithm")
                standard = tf.random_uniform((1,), 0, 1, dtype=tf.float32)[0]

                # x, y축 으로 늘릴 양
                random_height = tf.random_uniform(shape=(1,), minval=128, maxval=257, dtype=tf.int32)[0]
                random_width = tf.random_uniform(shape=(1,), minval=128, maxval=257, dtype=tf.int32)[0]

                image_resized, label_resized = tf.cond(tf.greater_equal(standard, 0.7), \
                                                       lambda: (tf.image.resize_images(images=image,
                                                                                       size=(tf.shape(image)[
                                                                                                 0] + random_height,
                                                                                             tf.shape(image)[
                                                                                                 1] + random_width),
                                                                                       method=1),
                                                                tf.image.resize_images(images=label,
                                                                                       size=(tf.shape(label)[
                                                                                                 0] + random_height,
                                                                                             tf.shape(label)[
                                                                                                 1] + random_width),
                                                                                       method=1)), \
                                                       lambda: (image, label))
                # 원 사이즈로 자르기
                concat_resized = tf.concat(values=[image_resized, label_resized], axis=-1)
                concat_cropped = tf.random_crop(concat_resized, size=(
                    tf.shape(image)[0], tf.shape(image)[1], tf.shape(concat_resized)[-1]))
                image, label = tf.split(concat_cropped, [tf.shape(image)[-1], tf.shape(label)[-1]], axis=-1)
        else:
            print("\nAugmentation Algorithm nothing\n")

        return image, label, file_name

    # tf.data.TFRecordDataset를 사용하는 방법 - TRRecord(이진 파일, 직렬화된 입력 데이터)라는 텐서플로우 표준 파일형식으로 저장된 파일을 불러와서 처리하기
    def Using_TFRecordDataset(self):

        dataset = tf.data.TFRecordDataset(self.TFRecord_path)

        # TFRecord에서 이미지 읽어오기
        dataset = dataset.map(self._image_preprocessing, num_parallel_calls=10)

        if self.use_TrainDataset:
            # 확률적 Augmentation 알고리즘 적용
            dataset = dataset.map(self._data_augmentation, num_parallel_calls=10)
            dataset = dataset.repeat().batch(self.batch_size).shuffle(buffer_size=10)
            dataset = dataset.prefetch(buffer_size=self.batch_size)
        else:
            dataset = dataset.repeat().batch(self.batch_size)
            dataset = dataset.prefetch(buffer_size=self.batch_size)

        iterator = dataset.make_one_shot_iterator()

        # tf.python_io.tf_record_iterator는 무엇인가 ? TFRecord 파일에서 레코드를 읽을 수 있는 iterator이다.
        return iterator.get_next(), sum(1 for _ in tf.python_io.tf_record_iterator(self.TFRecord_path))

    def iterator(self):

        next_batch, db_length = self.Using_TFRecordDataset()

        return next_batch, db_length


if __name__ == "__main__":

    dataset = Dataset(batch_size=1, use_TrainDataset=True, input_range="0~1",
                      Augmentation_algorithm=0)
    iterator, next_batch, data_length = dataset.iterator()

else:
    print("Dataset imported")
