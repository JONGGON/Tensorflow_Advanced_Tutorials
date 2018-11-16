from tensorflow.python.client import device_lib
#import os
import ImageToImageTranslation as pix2pix

'''
1. 설명
데이터셋 다운로드는 - https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/ 에서 <데이터셋> 을 내려 받자. 
논문에서 추천하는 hyperparameter 는 200 epoch, 1 ~ 10 batch size 정도, beta1 = 0.5, beta2=0.999, lr=0.0002
입력 크기 : <256x256x3>
출력 크기 : <256x256x3>
저자코드 - https://github.com/phillipi/pix2pix/blob/master/models.lua
generator는 unet을 사용한다.
discriminator의 구조는 PatchGAN 70X70을 사용한다. 

Training, Test Generator의 동작 방식이 같다. - 드롭아웃 적용, batchnorm 이동 평균 안씀 - 그래도 옵션으로 주자
At inference time, we run the generator net in exactly
the same manner as during the training phase. This differs
from the usual protocol in that we apply dropout at test time,
and we apply batch normalization [28] using the statistics of
the test batch, rather than aggregated statistics of the training
batch. This approach to batch normalization, when the
batch size is set to 1, has been termed “instance normalization”
and has been demonstrated to be effective at image
generation tasks [53]. In our experiments, we use batch
sizes between 1 and 10 depending on the experiment

논문 내용과 거의 똑같이 구현했다. + random crop을 랜덤한 크기로 적용했다.

2. loss에 대한 옵션
distance_loss = 'L1' 일 경우 , generator에서 나오는 출력과 실제 출력값을 비교하는 L1 loss를 생성
distance_loss = 'L2' 일 경우 , generator에서 나오는 출력과 실제 출력값을 비교하는 L2 loss를 생성
distamce_loss = None 일 경우 , 추가적인 loss 없음

3. pathGAN?
patchGAN 은 논문의 저자가 붙인이름이다. - ReceptiveField 에 대한 이해가 필요하다. -> 이 내용에 대해 혼돈이 있을 수 있으니 ref폴더의 안의 receptiveField 내용과 receptiveFieldArithmetic폴더의 receptiveField 크기 구현 코드를 참고한다.
저자의 답변이다.(깃허브의 Issues에서 찾았다.)
This is because the (default) discriminator is a "PatchGAN" (Section 2.2.2 in the paper).
This discriminator slides across the generated image, convolutionally, 
trying to classify if each overlapping 70x70 patch is real or fake. 
This results in a 30x30 grid of classifier outputs, 
each corresponding to a different patch in the generated image.
'''

'''
텐서플로우의 GPU 메모리정책 - GPU가 여러개 있으면 기본으로 메모리 다 잡는다. -> 모든 GPU 메모리를 다 잡는다. - 원하는 GPU만 쓰고 싶은 해결책은 CUDA_VISIBLE_DEVICES 에 있다. - 83번째 라인을 읽어라!!!
Allowing GPU memory growth
By default, TensorFlow maps nearly all of the GPU memory of all GPUs (subject to CUDA_VISIBLE_DEVICES) visible to the process.
This is done to more efficiently use the relatively precious GPU memory resources on the devices by reducing memory fragmentation.

In some cases it is desirable for the process to only allocate a subset of the available memory, 
or to only grow the memory usage as is needed by the process. TensorFlow provides two Config options on the Session to control this.

The first is the allow_growth option, which attempts to allocate only as much GPU memory based on runtime allocations: 
it starts out allocating very little memory, and as Sessions get run and more GPU memory is needed, we extend the GPU memory region needed by the TensorFlow process. 
Note that we do not release memory, since that can lead to even worse memory fragmentation. To turn this option on, set the option in the ConfigProto by:

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config, ...)
The second method is the per_process_gpu_memory_fraction option, which determines the fraction of the overall amount of memory that each visible GPU should be allocated. 
For example, you can tell TensorFlow to only allocate 40% of the total memory of each GPU by:

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.Session(config=config, ...)
'''

print("<<< * 한대의 컴퓨터에 여러대의 GPU 가 설치되어 있을 경우 참고할 사항 >>>")
print(
    "<<< 경우의 수 1 : GPU가 여러대 설치 / 통합개발 환경에서 실행 / GPU 번호 지정 원하는 경우 -> os.environ[\"CUDA_VISIBLE_DEVICES\"]=\"번호\"와 os.environ[\"CUDA_DEVICE_ORDER\"]=\"PCI_BUS_ID\"를 tf API를 하나라도 사용하기 전에 작성해 넣으면 됨 >>>")
print(
    "<<< 경우의 수 2 : GPU가 여러대 설치 / 터미널 창에서 실행 / GPU 번호 지정 원하는 경우  -> CUDA_VISIBLE_DEVICES = 0(gpu 번호) python main,py 을 터미널 창에 적고 ENTER - Ubuntu에서만 동작 >>>")
print("<<< CPU만 사용하고 싶다면? '현재 사용 가능한 GPU 번호' 에 없는 번호('-1'과 같은)를 적어 넣으면 됨 >>>\n")

# 특정 GPU로 학습 하고 싶을때, 아래의 2줄을 꼭 써주자.(Ubuntu , window 둘 다 가능) - 반드시 Tensorflow의 API를 하나라도 쓰기 전에 아래의 2줄을 입력하라!!!
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

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
DB_name 은 아래에서 하나 고르자
1. "cityscapes"
2. "facades"
3. "maps"
AtoB -> A : image,  B : segmentation
AtoB = True  -> image -> segmentation
AtoB = False -> segmentation -> image
'''
# 256x256 크기 이상의 다양한 크기의 이미지를 동시 학습 하는 것이 가능하다.(256 X 256으로 크기 제한을 뒀다.)
# -> 단 batch_size =  1 일 때만 가능하다. - batch_size>=2 일때 여러사이즈의 이미지를 동시에 학습 하고 싶다면, 각각 따로 사이즈별로 Dataset을 생성 후 학습시키면 된다.
# pix2pix GAN이나, Cycle gan이나 데이터셋 자체가 같은 크기의 이미지를 다루므로, 위 설명을 무시해도 된다.
# TEST=False 시 입력 이미지의 크기가 256x256 미만이면 강제 종료한다.
# TEST=True 시 입력 이미지의 크기가 256x256 미만이면 강제 종료한다.
# optimizers_ selection = "Adam" or "RMSP" or "SGD"
pix2pix.model(DB_name="facades",
              TEST=True,  # TEST=False -> Training or TEST=True -> TEST
              # 대량의 데이터일 경우 TFRecord=True가 더 빠르다.
              TFRecord=False,  # TFRecord=True -> TFRecord파일로 저장한후 사용하는 방식 사용 or TFRecord=False -> 파일에서 읽어오는 방식 사용
              AtoB=False,  # 데이터 순서 변경(ex) AtoB=True : image -> segmentation / AtoB=False : segmetation -> image)
              Inputsize_limit=(256, 256),  # 입력되어야 하는 최소 사이즈를 내가 지정 - (256,256) 으로 하자
              filter_size=32,  # generator와 discriminator의 처음 layer의 filter 크기
              norm_selection="BN",  # IN - instance normalizaiton , BN -> batch normalization, NOTHING
              regularizer=" ", # L1 or L2 정규화 -> 오버피팅 막기 위함
              scale=0.0001, # L1 or L2 정규화 weight
              Dropout_rate=0.5,  # generator의 Dropout 비율
              distance_loss=" ",  # L2 or NOTHING
              distance_loss_weight=100,  # distance_loss의 가중치
              optimizer_selection="Adam",  # optimizers_ selection = "Adam" or "RMSP" or "SGD"
              beta1=0.5, beta2=0.999,  # for Adam optimizer
              decay=0.999, momentum=0.9,  # for RMSProp optimizer
              image_pool=False,  # discriminator 업데이트시 이전에 generator로 부터 생성된 이미지의 사용 여부
              image_pool_size=50,  # image_pool=True 라면 몇개를 사용 할지?
              learning_rate=0.0002, training_epochs=1, batch_size=1, display_step=1,
              inference_size=(256, 256),  # TEST=True 일때, inference 할 수 있는 최소의 크기를 256 x 256으로 크기 제한을 뒀다.
              using_moving_variable=False,  # TEST=True 일때, Moving Average를 Inference에 사용할지 말지 결정하는 변수
              only_draw_graph=False,  # TEST=False 일 때 only_draw_graph=True이면 그래프만 그리고 종료한다.
              show_translated_image=True,  # TEST=True 일 때 변환된 이미지를 보여줄지 말지
              weights_to_numpy=False,  # TEST=True 일 때 가중치를 npy 파일로 저장할지 말지
              save_path="translated_image")  # TEST=True 일 때 변환된 이미지가 저장될 폴더
