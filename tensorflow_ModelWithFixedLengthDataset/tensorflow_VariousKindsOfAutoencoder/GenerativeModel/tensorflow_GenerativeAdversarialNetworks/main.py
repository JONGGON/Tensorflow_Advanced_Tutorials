import GenerativeAdversarialNetworks

# optimizers_ selection = "Adam" or "RMSP" or "SGD"
# 원하는 숫자를 생성이 가능하다.
# 흑백 사진(Generator의 입력으로) -> 컬러(Discriminator의 입력)로 만들기
# 선화(Generator의 입력)를 (여기서 채색된 선화는 Discriminator의 입력이 된다.)채색가능
'''
targeting = False 일 때는 숫자를 무작위로 생성하는 GAN MODEL 생성 - General GAN
targeting = True 일 때는 숫자를 타게팅 하여 생성하는 GAN MODEL 생성 - Conditional GAN 

targeting = True 일 때 -> distance_loss = 'L1' 일 경우 , generator에서 나오는 출력과 실제 출력값을 비교하는 L1 loss를 생성
targeting = True 일 때 -> distance_loss = 'L2' 일 경우 , generator에서 나오는 출력과 실제 출력값을 비교하는 L2 loss를 생성
targeting = True 일 때 -> distamce_loss = None 일 경우 , 추가적인 loss 없음
'''
# batch normalization은 Hidden Layer에만 추가합니다. 또한 활성화 함수전에 적용합니다.
# regularization -> batch_norm = False 일때, L2 or L1 or nothing
GenerativeAdversarialNetworks.model(TEST=True, noise_size=128, targeting=True, distance_loss="L1",
                                    distance_loss_weight=1, \
                                    optimizer_selection="Adam", learning_rate=0.0002, training_epochs=50,
                                    batch_size=128,
                                    display_step=1, regularization='L2', scale=0.0001)
