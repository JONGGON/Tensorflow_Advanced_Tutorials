from tensorflow.python.client import device_lib

from DDQN import *

'''
window 10 에서도 된다.
pip install gym
pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py # 0.1.1 버전인 atari 이지만 있을 것은 다 있다.
pip install gym[atari]
'''

# if platform.system() == "Linux" or platform.system() == "mac":
#     print("<<< 현재 운영체제 :{} -> 실행 가능 >>>".format(platform.system()))
#     pass
# else:
#     print("<<<현재 운영체제 :{} -> 실행 불가능 >>>".format(platform.system()))
#     print("<<< 강제 종료 >>>")
#     exit()

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

'''
구현시 주의 할 점(+ 짜증나는 점 : 학습이 너무 오래 걸린다.)

1. 시계열 입력 : 공간이 정해져 있고 입력이 한칸씩 왼쪽으로 밀리면 맨 오른쪽에 새로운 '관측'이 추가되는 구조 
2. 재현메모리 : 구현이 약간 까다로웠음 / 재현메모리에 들어가는 데이터의 크기는 UINT8 data type이 효율적이며, 실제 학습 데이터로 쓰일 때는 0~1 사이로 정규화 되서 들어 가야 한다.(이게 안되면 학습이 잘 안되는 것 같다.)
3. Deterministic ? openai gym에서 게임의 모든 frame을 실행하지 않고 4frame을 건너뛰어서 실행하는 버전!!! 모든 게임의 프레임을 다 볼 필욘 없지 않겠는가???(논문참고^^^)
4. save_step, copy_step은 학습이 각각 save_step, copy_step만큼 진행될때 마다 업데이트 되게 하는 변수이다. - training_step은 agent가 총 움직일 횟수를 의미하는 것일 뿐.
   - 사실 epoch당 1 episode(게임 한판) 구조로도 구현 할 수 있다.(자기 맘이지^^^) 
5. Test Code에서도 epsilon-greedy 정책을 사용해야한다는 사실!!! 
'''

Atari = model(
    # https://gym.openai.com/envs/#atari
    # ex) TennisDeterministic-v0, PongDeterministic-v4, BattleZoneDeterministic-v4, BreakoutDeterministic-v4
    model_name="PongDeterministic-v4",
    training_display=(True, 10000),
    training_step=200000000,
    training_start_point=10000,
    # 4번마다 한번씩만 학습 하겠다는 것이다.
    # -> 4번중 3번은 게임을 진행해보고 4번째에는 그 결과들을 바탕으로 학습을 하겠다는 이야기
    training_interval=4,
    rememorystackNum=300000,
    save_step=10000,  # 가중치 업데이트의 save_step 마다 저장한다.
    copy_step=10000,  # 가중치 업데이트의 copy_step 마다 저장한다.
    framesize=4,  # 입력 상태 개수
    learning_rate=0.00025,
    momentum=0.95,
    egreedy_max=1,
    egreedy_min=0.1,
    egreedy_step=1000000,
    discount_factor=0.99,
    batch_size=32,
    with_replacement=True,  # True : 중복추출, False : 비중복 추출
    only_draw_graph=False,  # model 초기화 하고 연산 그래프 그리기
    SaveGameMovie=True)

Atari.train  # 학습 하기
Atari.test  # 테스트 하기
