# import argparse

from tensorflow.python.client import device_lib

from test import *
from train import Model

print("<<< * 한대의 컴퓨터에 여러대의 GPU 가 설치되어 있을 경우 참고할 사항 >>>")
print(
    "<<< 경우의 수 1 : GPU가 여러대 설치 / 통합개발 환경에서 실행 / GPU 번호 지정 원하는 경우 -> os.environ[\"CUDA_VISIBLE_DEVICES\"]=\"번호\"와 os.environ[\"CUDA_DEVICE_ORDER\"]=\"PCI_BUS_ID\"를 tf API를 하나라도 사용하기 전에 작성해 넣으면 됨 >>>")
print(
    "<<< 경우의 수 2 : GPU가 여러대 설치 / 터미널 창에서 실행 / GPU 번호 지정 원하는 경우  -> CUDA_VISIBLE_DEVICES=0(gpu 번호) python main.py 을 터미널 창에 적고 ENTER - Ubuntu에서만 동작 >>>")
print("<<< CPU만 사용하고 싶다면? '현재 사용 가능한 GPU 번호' 에 없는 번호('-1'과 같은)를 적어 넣으면 됨 >>>\n")

# 특정 GPU로 학습 하고 싶을때, 아래의 2줄을 꼭 써주자.(Ubuntu , window 둘 다 가능) - 반드시 Tensorflow의 API를 하나라도 쓰기 전에 아래의 2줄을 입력하라!!!
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# 현재 사용하고 있는 GPU 번호를 얻기 위한 코드 - 여러개의 GPU를 쓸 경우 정보 확인을 위해!
print("<<< Ubuntu Terminal 창에서 지정해준 경우, 무조건 GPU : 1대, GPU 번호 : 0 라고 출력 됨 >>>")
local_device_protos = device_lib.list_local_devices()
GPU_List = [x.name for x in local_device_protos if x.device_type == 'GPU']
# gpu_number_list = []
print("<<< # 사용 가능한 GPU : {} 대 >>>".format(len(GPU_List)))
print("<<< # 사용 가능한 GPU 번호 : >>>", end="")
for i, GL in enumerate(GPU_List):
    num = GL.split(":")[-1]
    # gpu_number_list.append(num)
    if len(GPU_List) - 1 == i:
        print(" " + num)
    else:
        print(" " + num + ",", end="")

# parser = argparse.ArgumentParser()
# '''
# train = True  --> 학습
# train = False --> 평가
# '''
# parser.add_argument("--train", help="bool type", type=bool, default=False)
# args = parser.parse_args()
Train = True
#if args.train:
if Train:
    model = Model(DB_choice_percentage=1,  # 0=< DB_choice_percentage <= 1
                  Inputsize_limit=(256, 256),  # 입력되어야 하는 최소 사이즈를 내가 지정 - (256,256) 으로 하자
                  input_range="0~1", # or -1~1
                  network="UNET",  # UNET
                  variable_scope="Segmentation",
                  init_filter_size=32,  # generator와 discriminator의 처음 layer의 filter 크기
                  regularizer=" ",  # L1 or L2 정규화 -> 오버피팅 막기 위함
                  scale=0.000001,  # L1 or L2 정규화 weight
                  Dropout_rate=0.5,  # generator의 Dropout 비율
                  loss_selection="pixelwise_softmax_cross_entropy",  # or soft_dice or squared_soft_dice
                  loss_weight=1,  # loss의 가중치
                  # distance_loss의 가중치  / distance_loss="ALL" -> distance_loss_weight*L1 + (1-distance_loss_weight)*L2
                  optimizer_selection="Adam",
                  # optimizers_ selection = "Adam" or "RMSP" or "SGD or Momentum or Nesterov"
                  # optimizers_ selection = "Adam" or "RMSP" or "SGD" or "Momentum" or "Nesterov"
                  beta1=0.9, beta2=0.999,  # for Adam optimizer
                  decay=0.999, momentum=0.9,  # for RMSProp optimizer
                  # Augmentation Algorithm
                  # 1. random padding - batch size > 1일 때는 학습되지 않는다.
                  # 2. -90, -180, -270 rotation 확률적 적용
                  # 3. -10 ~ 10 rotation 확률적 적용
                  # 4. -10 ~ 10 상하좌우 translation 확률적 적용
                  # 5. 원본데이터의 가로 + (128 ~ 256), 원본데이터의 세로 + (128 ~ 256) 한 후 원본이미지로 random crop 확률적 적용
                  Augmentation_algorithm=5,
                  learning_rate=0.0002,  # 학습률
                  lr_decay_epoch=200,  # 몇 epoch 뒤에 learning_rate를 줄일지
                  lr_decay=1,  # learning_rate를 얼마나 줄일지
                  learning_time=100,  # 몇 개의 DB마다 시간을 보여줄지
                  training_epochs=200,  # 총 학습 횟수
                  batch_size=1,
                  save_step=10,  # 몇 epoch마다 가중치를 저장할지
                  using_latest_weight=True,  # 최근의 가중치를 쓸건지? 말건지
                  WeightSelection=500,  # using_latest_weight=False 일 때 수동으로 내가 선택한 가중치를 선택해서 결과를 출력한다.
                  Accesskey="way",  # graph 저장할 때 변수들의 통로로 사용할 key - GRPC를 위해 만들어 놓음
                  only_draw_graph=False,  # TEST=False 일 때 only_draw_graph=True이면 그래프만 그리고 종료한다
                  class_number=2)

    model.train()
else:
    db_test(model_name="32UNETSCELIN0~1",
            Inputsize_limit=(256, 256),
            weights_to_numpy=True,
            using_latest_weight=True,
            WeightSelection=250,
            Dropout_rate=1,
            masking=True,
            masking_Image_save_path="masking_image",
            HTML_Report=True,
            HTML_Font_Size=30)
