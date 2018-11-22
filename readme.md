>## **TensorFlow Advanced Tutorials**
        
* ### **Topics** 

    * **니킬부두마의 **딥러닝의 정석**에서 소개하는 내용들과 개인적으로 공부한 내용들에 대해 정리하며 작성한 코드들입니다.**  

    * 모든 코드는 main.py 에서 실행합니다.
        ```cmd
        # argparse는 사용하지 않습니다.
        1. cmd(window) or Terminal(Ubuntu)에서 실행하는 경우 python main.py을 입력 후 실행합니다. 

        2. IDE(pycharm, vscode and so on.)에서 실행하는 경우 main.py를 실행합니다.  
        ```


    * ### **Model With Fixed Length Dataset**
        
        * [***Fully Connected Layer***](https://github.com/JONGGON/Tensorflow_Advanced_Tutorials/tree/master/tensorflow_ModelWithFixedLengthDataset/tensorflow_FullyConnectedNeuralNetwork)
            * 기본적인 FullyConnected Neural Network(전방향 신경망) 입니다.

        * [***Convolution Neural Network***](https://github.com/JONGGON/Tensorflow_Advanced_Tutorials/tree/master/tensorflow_ModelWithFixedLengthDataset/tensorflow_ConvolutionNeuralNetwork)

            * 기본적인 Convolution Neural Network(합성곱 신경망) 입니다.
            
            * Convolution 층으로만 구성합니다.(마지막 층에 1x1 filter를 사용합니다.)
            
            * [ReceptiveField(수용 영역)크기 계산법을 사용해서 네트워크의 구조를 결정 했습니다.](https://github.com/JONGGON/Tensorflow_Advanced_Tutorials/blob/master/tensorflow_ModelWithFixedLengthDataset/tensorflow_ConvolutionNeuralNetwork/ReceptiveField_inspection/rf.py)

         * **Various Kinds Of Autoencoder**
            * **Feature Extraction Model**
                * [***Autoencoder And PCA***](https://github.com/JONGGON/Tensorflow_Advanced_Tutorials/tree/master/tensorflow_ModelWithFixedLengthDataset/tensorflow_VariousKindsOfAutoencoder/FeatureExtractionModel/tensorflow_AutoencoderAndPCA)
                    * 기본적인 Autoencoder 를 PCA 와 비교한 코드입니다.

                * [***Denoising Autoencoder And PCA***](https://github.com/JONGGON/Tensorflow_Advanced_Tutorials/tree/master/tensorflow_ModelWithFixedLengthDataset/tensorflow_VariousKindsOfAutoencoder/FeatureExtractionModel/tensorflow_DenoisingAutoencoderAndPCA)
                    * 네트워크의 복원 능력을 강화하기 위해 입력에 노이즈를 추가한 Denoising Autoencoder 를 PCA 와 비교한 코드입니다.

                * [***SparseAutoencoder And PCA***](https://github.com/JONGGON/Tensorflow_Advanced_Tutorials/tree/master/tensorflow_ModelWithFixedLengthDataset/tensorflow_VariousKindsOfAutoencoder/FeatureExtractionModel/tensorflow_SparseAutoencoderAndPCA)
                    * 소수의 뉴런만 활성화 되는 Sparse Autoencoder 입니다.
            * **Generative Model**

                * [***Basic and Conditional Generative Adversarial Networks***](https://github.com/JONGGON/Tensorflow_Advanced_Tutorials/tree/master/tensorflow_ModelWithFixedLengthDataset/tensorflow_VariousKindsOfAutoencoder/GenerativeModel/tensorflow_GenerativeAdversarialNetworks)
                    * 무작위로 데이터를 생성해내는 GAN 과 네트워크에 조건을 걸어 원하는 데이터를 생성하는 조건부 GAN 입니다.

                * [***Basic and Conditional Variational Autoencoder***](https://github.com/JONGGON/Tensorflow_Advanced_Tutorials/tree/master/tensorflow_ModelWithFixedLengthDataset/tensorflow_VariousKindsOfAutoencoder/GenerativeModel/tensorflow_VariationalAutoencoder)
                    * Autoencoder를 생성모델로 사용합니다. 짧게 줄여 VAE라고 합니다. 중간층의 평균과 분산에 따라 무작위로 데이터를 생성하는 VAE 와 중간층의 평균과 분산에 target정보를 주어 원하는 데이터를 생성하는 VAE 입니다.
         * **Application**
            * [***Tensorboard Embedding Visualization***](https://github.com/JONGGON/Tensorflow_Advanced_Tutorials/tree/master/tensorflow_Application/tensorflow_MnistEmbeddingVisualizationWithTensorboard)
                ```
                주의 : Tensorboard event 파일과 weight 파일은 같은 경로에 있어야 합니다. 
                ```
                * Tensorboard Embedding에서 지원하는 PCA, T-SNE와 같은 차원 축소 알고리즘으로 다양한 데이터들을 시각화 할 수 있습니다. 여기서는 MNIST 데이터를 시각화 합니다.

            * [***LottoNet***](https://github.com/JONGGON/Tensorflow_Advanced_Tutorials/tree/master/tensorflow_Application/tensorflow_AutoencoderLottoNet)
                * 로또 당첨의 꿈을 이루고자 전방향 신경망을 사용해 단순히 로또 번호를 예측하는 코드입니다.
                * 네트워크 Graph 구조를 담은 meta 파일을 저장하고 불러오는 코드가 포함되어 있습니다. tensorflow.add_to_collection, tensorflow.get_collection 를 사용합니다.
                * tf.data.Dataset를 사용합니다. 자신의 데이터를 학습 네트워크에 적합한 형태로 쉽게 처리 할 수 있게 도와주는 API입니다.
                    * Tensorflow를 처음 시작하는 대부분의 사람들은 Tensorflow가 자체 처리해서 제공하는 MNIST, CIFAR 같은 Toy 데이터들을 사용해 여러 가지 학습 알고리즘들을 공부하게 됩니다. 어느 정도 학습이 진행되면 '나'만의 데이터를 사용해서 어떤 네트워크를 학습시켜보고 싶은데... 어떻게 처리 해야 하지? 조금 쉽게 하는 방법 없나? 라는 의문이 듭니다. 이 의문에 대한 답이 'tf.data.Dataset'에 있습니다.
                * tf.train.Saver().export_meta_graph API 와 tf.train.import_meta_graph API를 사용하여 Training, Test 코드를 각각 실행합니다.
            * [***Neural Style***](https://github.com/JONGGON/Tensorflow_Advanced_Tutorials/tree/master/tensorflow_Application/tensorflow_NeuralStyle)
                * 내 사진을 예술 작품으로 바꿔주는 유명한 논문인 "A Neural Algorithm of Artistic Style" 의 구현 입니다.
            * [***Word2Vector SkipGram With TSNE***](https://github.com/JONGGON/Tensorflow_Advanced_Tutorials/tree/master/tensorflow_Application/tensorflow_Word2Vector_SkipGram_WithTSNE)
                * 아무런 관계가 없는 것처럼 표현(one-hot encoding)된 단어들을 낮은 차원의 벡터로 표현함과 동시에 단어간의 관계를 표현하는 방법입니다. Word2Vector에는 CBOW모델과 Skip-Gram 모델이 있습니다. 여기서는 Skip-Gram 모델을 구현합니다.
                * tf.train.Saver().export_meta_graph API 와 tf.train.import_meta_graph API를 사용하여 Training, Test 코드를 각각 실행합니다.(그래프파일인 meta파일을 저장하여 Test시 불러옵니다.)
                * Tensorboard Embedding에서 지원하는 PCA, T-SNE와 같은 차원 축소 알고리즘으로 Embedding된 단어들의 관계를 시각화 해봅니다.

            * [***Image To Image Translation With Conditional Adversarial Networks***](https://github.com/JONGGON/Tensorflow_Advanced_Tutorials/tree/master/tensorflow_Application/tensorflow_ImageToImageTranslationWithConditionalAdversarialNetworks_Graph)
                * 어떤 도메인의 이미지의 다른 도메인의 이미지로의 변환이라는 거룩한 목적을 위해 고안된 네트워크입니다. ConditionalGAN 과 UNET을 사용하여 네트워크 구성 합니다.
                * 네트워크 구조 및 학습 방법은 논문에서 제시한 내용과 거의 같습니다.(Discriminator 구조인 PatchGAN 의 크기는 70X70 입니다. - [ReceptiveField 계산법](https://github.com/JONGGON/Tensorflow_Advanced_Tutorials/blob/master/tensorflow_ModelWithFixedLengthDataset/tensorflow_ConvolutionNeuralNetwork/ReceptiveField_inspection/rf.py)
                * 2가지의 데이터 전처리 방법  -
                    1. 효율적인 데이터 전처리를 위해 tf.data.Dataset을 사용할 수 있습니다.(tf.data.Dataset로 처리된 데이터를 그래프 구조 안에 포함시킵니다.)
                    2. 더 효율적인 데이터 전처리를 위해 TFRecord 형식으로 데이터를 저장하고  tf.data.TFRecordDataset 를 사용할 수 있습니다.
                * tf.train.Saver().export_meta_graph API 와 tf.train.import_meta_graph API를 사용하여 Training, Test 코드를 각각 실행합니다.(그래프파일인 meta파일을 저장하여 Test시 불러옵니다.)
                * 256x256 크기 이상의 서로 다른 크기의 이미지들을 동시에 학습 할 수 있습니다.
                    * 256x256 크기의 이미지로 학습한 생성 네트워크에 512x512 크기의 이미지를 입력으로 넣어 성능을 평가하기 위한 기능입니다. 다양한 크기의 이미지를 동시에 학습하는 것도 가능합니다. ( 관련 내용 : [Image To Image Translation With Conditional Adversarial Networks Using edges2shoes Dataset 논문의 7p 참고](https://arxiv.org/pdf/1611.07004.pdf))
                * CycleGan에서 사용하는 ImagePool 함수(batch size가 1일 때만 동작)도 추가했습니다.
                    * to reduce model oscillation [14], we follow
                    Shrivastava et al’s strategy [45] and update the discriminators using a history of generated images rather than the ones produced by the latest generative networks. We keep an image buffer that stores the 50 previously generated images.
            
            * [***Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks***](https://github.com/JONGGON/Tensorflow_Advanced_Tutorials/tree/master/tensorflow_Application/tensorflow_UnpairedImageToImageTranslationUsingCycleConsistentAdversarialNetworks_Graph)
                * Image To Image Translation With Conditional Adversarial Newort을 학습시키기 위해선 입력과 출력이 한 쌍인 데이터가 필요했습니다. 그러나 이 논문에서 제시한 방법은 입력과 출력이 쌍일 필요가 없습니다.
                * CycleGan은 전혀 다른 도메인에 있는 두 종류의 데이터 집단을 자연스럽게 이어주는 연결고리를 찾아내는 과정이라고 생각합니다.
                * ImagePool 함수는 batch size가 1일 때만 동작합니다.
                * 네트워크 구조 및 학습 방법은 논문에서 제시한 내용과 거의 같습니다.
                * 2가지의 데이터 전처리 방법  
                    1. 효율적인 데이터 전처리를 위해 tf.data.Dataset을 사용할 수 있습니다.(tf.data.Dataset로 처리된 데이터를 그래프 구조 안에 포함시킵니다.)
                    2. 더 효율적인 데이터 전처리를 위해 TFRecord 형식으로 데이터를 저장하고  tf.data.TFRecordDataset 를 사용할 수 있습니다.
                * tf.train.Saver().export_meta_graph API 와 tf.train.import_meta_graph API를 사용하여 Training, Test 코드를 각각 실행합니다.(그래프파                * 다른 환경에 대해서도 가능합니다.(참고 : https://gym.openai.com/envs/#atari) 일인 meta파일을 저장하여 Test                * 다른 환경에 대해서도 가능합니다.(참고 : https://gym.openai.com/envs/#atari)  불러옵니다.)
                * 256x256 크기 이상의 서                * 다른 환경에 대해서도 가능합니다.(참고 : https://gym.openai.com/envs/#atari)  다른 크기의 이미지들을 동시에 학습 할 수 있습니다.
                    * 256x256 크기의 이미지로 학습한 생성 네트워크에 512x512 크기의 이미지를 입력으로 넣어 성능을 평가하기 위한 기능입니다. 다양한 크기의 이미지를 동시에 학습하는 것도 가능합니다. ( 관련 내용 : [Image To Image Translation With Conditional Adversarial Networks Using edges2shoes Dataset 논문의 7p 참고](https://arxiv.org/pdf/1611.07004.pdf))

            * [***Policy Gradient with CartPole***](https://github.com/JONGGON/Tensorflow_Advanced_Tutorials/tree/master/tensorflow_ReinForcementLearning/PG_CartPole) 
                * 카트 위에 놓인 막대가 넘어지지 않도록 왼쪽 또는 오른쪽으로 가속시키는 2D 시뮬레이션입니다.
                * 일단 에피소드를 몇 번 진행해보고 이를 평균내어 학습합니다.
                    * 몬테카를로 정책 그라디언트(Monte Carlo Policy Gradient) 방법입니다. 
                    
            * [***Double Q-learning***](https://github.com/JONGGON/Tensorflow_Advanced_Tutorials/tree/master/tensorflow_ReinForcementLearning/Double_DQN)
                * Double Q-네트워크(Deep Q-networks)를 사용한 방법입니다.
                * 다양한 gym 환경들에 대해서 학습이 가능합니다.(참고 : https://gym.openai.com/envs/#atari) 

>## **개발 환경**
* os : ```window 10.1 64bit, ubuntu 18.04``` 
* python version(`3.6.4`) : `anaconda3 4.4.10` 
* IDE : `pycharm Community Edition 2018.1.2`
    
>## **코드 실행에 필요한 파이썬 모듈** 
* Tensorflow-1.10
* numpy, collections, pandas, random
* matplotlib, scikit-learn, opencv-python, scipy, copy
* tqdm, os, glob, shutil, urllib, zipfile, tarfile

>## **Author**
Kim Jong Gon

medical18@naver.com
