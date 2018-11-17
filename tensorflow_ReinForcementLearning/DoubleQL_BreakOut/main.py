import platform

from tensorflow.python.client import device_lib

from DDQN import *

if platform.system() == "Linux" or platform.system() == "mac":
    print("<<< 현재 운영체제 :{} -> 실행 가능 >>>".format(platform.system()))
    pass
else:
    print("<<<현재 운영체제 :{} -> 실행 불가능 >>>".format(platform.system()))
    print("<<< 강제 종료 >>>")
    exit()

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


Atari = model(
    # https://gym.openai.com/envs/#atari
    # ex) Tennis-v0, Pong-v0, BattleZone-v0
    model_name="Breakout-v0",
    training_display=True,
    SaveGameMovie=True,
    training_step=200000000,
    training_interval=4,
    rememorystackNum=500000, #메모리 문제로 줄인다.
    save_step=30000,
    copy_step=10000,

    framesize=4, # 입력 상태 개수
    learning_rate=0.00025,
    momentum = 0.95,
    egreedy_max=1,
    egreedy_min=0.1,
    egreedy_step=1000000,
    discount_factor=0.99,
    batch_size=32,
    with_replacement=True,
    only_draw_graph=False)  # model 초기화 하고 연산 그래프 그리기

Atari.train  # 학습 하기
Atari.test  # 테스트 하기
