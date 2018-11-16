import collections
import os
import random
import urllib
import zipfile

import numpy as np
import tensorflow as tf

'''
https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/examples/tutorials/word2vec/word2vec_basic.py 의
코드를 수정
'''


class data_preprocessing(object):

    def __init__(self, url='http://mattmahoney.net/dc/', filename='text8.zip',
                 expected_bytes=31344016,
                 vocabulary_size=10000):

        self.url = url
        self.filename = filename
        self.expected_bytes = expected_bytes
        self.vocabulary_size = vocabulary_size
        self.maybe_download()
        self.data, self.count, self.dictionary, self.reverse_dictionary = self.build_dataset()

        # 전체 dataset의 현재 index 값을 표시하기 위한 변수 -> 하나씩 증가함
        self.data_index = 0

    def maybe_download(self):

        """Download a file if not present, and make sure it's the right size."""
        # Step 1: Download the data.
        if not os.path.exists(self.filename):
            print("<<< {} Dataset Download required >>>".format(self.filename))
            self.filename, _ = urllib.request.urlretrieve(self.url + self.filename, self.filename)
            print("<<< {} Dataset Download Completed >>>".format(self.filename))
        statinfo = os.stat(self.filename)
        if statinfo.st_size == self.expected_bytes:
            print('Found and verified', self.filename)
        else:
            print(statinfo.st_size)
            raise Exception(
                'Failed to verify ' + self.filename + '. Can you get to it with a browser?')

    def build_dataset(self):

        """Extract the first file enclosed in a zip file as a list of words"""
        with zipfile.ZipFile(self.filename) as f:
            original_words = tf.compat.as_str(f.read(f.namelist()[0])).split()

        count = [['UNK', -1]]
        count.extend(collections.Counter(original_words).most_common(self.vocabulary_size - 1))

        selected_dataset_dictionary = dict()
        for i, (select_word, _) in enumerate(count):
            selected_dataset_dictionary[select_word] = i

        data = list()
        unk_count = 0

        # self.vocabulary_size 개수 만큼만 데이터셋 구성으로 사용
        for or_word in original_words:
            if or_word in selected_dataset_dictionary:
                index = selected_dataset_dictionary[or_word]
            else:
                index = 0  # dictionary['UNK']
                unk_count += 1
            data.append(index)
        count[0][1] = unk_count
        reverse_dictionary = dict(zip(selected_dataset_dictionary.values(), selected_dataset_dictionary.keys()))
        return data, count, selected_dataset_dictionary, reverse_dictionary

    def generate_batch(self, batch_size=64, num_skips=2, window_size=1):

        '''
        batch_size => 한번에 학습할 학습 데이터의 수
        num_skips => 하나의 문장에서 몇개의 데이터를 생성할지 결정하는 변수
        '''

        assert batch_size % num_skips == 0
        assert num_skips <= 2 * window_size

        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

        span = 2 * window_size + 1  # if window_size=1, [ window_size target window_size ]

        '''
        maxlen = span의 의미? : span까지의 길이 만큼이 계속 유지
        '''

        buffer = collections.deque(maxlen=span)  # 길이가 span인 buffer생성

        for _ in range(span):
            buffer.append(self.data[self.data_index])  # buffer에 데이터 추가
            self.data_index = (self.data_index + 1) % len(self.data)

        target = window_size  # (타깃)입력
        context = window_size  # (문맥)출력

        for i in range(batch_size // num_skips):  # 몫 반환
            context_to_avoid = [context]
            for j in range(num_skips):

                # 아래의 3줄의 코드는 제대로된 데이터를 생성하는데 중요한 역할을 한다.
                while context in context_to_avoid:  # targets_to_avoid 안에 있는 값이 target이 아닐때까지 실행
                    context = random.randint(0, span - 1)
                context_to_avoid.append(context)  # 이미 (문맥)출력으로 사용된 값을 제외시키는 역할

                # 데이터 생성
                batch[i * num_skips + j] = buffer[target]
                labels[i * num_skips + j, 0] = buffer[context]

            # deque의 append 함수가 하는 일 : 새로운 데이터가 들어오면 맨 왼쪽인자(기존 데이터)부터 자동으로 삭제됨
            # 아래의 2줄의 코드는 self.data가 다음 데이터를 가리키게 하는 코드
            buffer.append(self.data[self.data_index])
            self.data_index = (self.data_index + 1) % len(self.data)

        return batch, labels


if __name__ == "__main__":
    # optimizers_ selection = "Adam" or "RMSP" or "SGD"
    dp = data_preprocessing(url='http://mattmahoney.net/dc/',
                            filename='text8.zip',
                            expected_bytes=31344016,
                            vocabulary_size=10000)
    batch_size = 64
    batch, labels = dp.generate_batch(batch_size=batch_size, num_skips=2, window_size=1)
    for i in range(batch_size):
        print(batch[i], dp.reverse_dictionary[batch[i]],
              '->', labels[i, 0], dp.reverse_dictionary[labels[i, 0]])
else:
    print("data_preprocessing.py imported")
