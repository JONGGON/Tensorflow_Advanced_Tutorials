import os
from collections import *

import cv2
import numpy as np

# class_number > 7 이상인 경우는 검은색으로 기본 설정하자.(이게 싫으면 색을 직접 추가하시길)
color = defaultdict(lambda: [0, 0, 0])

# segmentation용 color - 우선 최대 9개 색깔 지정
# BGR 순서
color[0] = [125, 237, 250]  # 노
color[1] = [0, 0, 255]  # 빨
color[2] = [255, 255, 255]  # 흰색
color[3] = [36, 130, 255]  # 주
color[4] = [0, 0, 0]  # 검
color[5] = [0, 255, 0]  # 초
color[6] = [255, 0, 0]  # 파
color[7] = [102, 0, 3]  # 남
color[8] = [217, 65, 128]  # 보


def draw_segmentation(masking_Image_save_path=None, file_name=None, input=None, target=None, prediction=None,
                      class_number=None, input_range=None):
    if not os.path.exists(masking_Image_save_path):
        os.makedirs(masking_Image_save_path)

    if input_range == "0~1":
        input = input * 255
    elif input_range == "-1~1":
        input = (input + 1) * 255

    # RGB -> BGR
    input = cv2.cvtColor(input.astype(np.uint8), cv2.COLOR_RGB2BGR)  # BGR 로 바꾼다

    # 영역별 색칠하기
    target = target[:, :, np.newaxis]
    prediction = prediction[:, :, np.newaxis]
    target_temp = np.zeros_like(np.tile(target, [1, 1, 3]))
    prediction_temp = np.zeros_like(np.tile(prediction, [1, 1, 3]))

    for i in range(class_number):
        target_temp += np.where(target == i, color[i], 0)
        prediction_temp += np.where(prediction == i, color[i], 0)

    # 입력 이미지 / 정답 이미지 / 예상 이미지
    concatenated_image = np.concatenate((input, target_temp, prediction_temp), axis=1).astype(np.uint8)
    concatenated_image = cv2.resize(concatenated_image, dsize=(1536, 512))
    cv2.imwrite(os.path.join(masking_Image_save_path, file_name+".jpg"), concatenated_image)
    print("<<< {} masking image 저장 완료 >>>".format(file_name))
