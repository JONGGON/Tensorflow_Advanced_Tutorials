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
                * tf.train.Saver().export_meta_graph API 와 tf.train.import_meta_graph API를 사용하여 Training, Test 코드를 각각 실행합니다.

            * [***Neural Style***](https://github.com/JONGGON/Tensorflow_Advanced_Tutorials/tree/master/tensorflow_Application/tensorflow_NeuralStyle)
                * 내 사진을 예술 작품으로 바꿔주는 유명한 논문인 "A Neural Algorithm of Artistic Style" 의 구현 입니다.

            * [***Word2Vector SkipGram With TSNE***](https://github.com/JONGGON/Tensorflow_Advanced_Tutorials/tree/master/tensorflow_Application/tensorflow_Word2Vector_SkipGram_WithTSNE)
                * 아무런 관계가 없는 것처럼 표현(one-hot encoding)된 단어들을 낮은 차원의 벡터로 표현함과 동시에 단어간의 관계를 표현하는 방법입니다. Word2Vector에는 CBOW모델과 Skip-Gram 모델이 있습니다. 여기서는 Skip-Gram 모델을 구현합니다.
                * tf.train.Saver().export_meta_graph API 와 tf.train.import_meta_graph API를 사용하여 Training, Test 코드를 각각 실행합니다.(그래프파일인 meta파일을 저장하여 Test시 불러옵니다.)
                * Tensorboard Embedding에서 지원하는 PCA, T-SNE와 같은 차원 축소 알고리즘으로 Embedding된 단어들의 관계를 시각화 해봅니다.

            * [***Image To Image Translation With Conditional Adversarial Networks***](https://github.com/JONGGON/Tensorflow_Advanced_Tutorials/tree/master/tensorflow_Application/tensorflow_ImageToImageTranslationWithConditionalAdversarialNetworks_Graph)
                * 자질구레한 문제가 생겨서 코드를 내립니다.
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
                * 자질구레한 문제가 생겨서 코드를 내립니다.
                * Image To Image Translation With Conditional Adversarial Newort을 학습시키기 위해선 입력과 출력이 한 쌍인 데이터가 필요했습니다. 그러나 이 논문에서 제시한 방법은 입력과 출력이 쌍일 필요가 없습니다.
                * CycleGan은 전혀 다른 도메인에 있는 두 종류의 데이터 집단을 자연스럽게 이어주는 연결고리를 찾아내는 과정이라고 생각합니다.
                * ImagePool 함수는 batch size가 1일 때만 동작합니다.
                * 네트워크 구조 및 학습 방법은 논문에서 제시한 내용과 거의 같습니다.
                * 2가지의 데이터 전처리 방법  
                    1. 효율적인 데이터 전처리를 위해 tf.data.Dataset을 사용할 수 있습니다.(tf.data.Dataset로 처리된 데이터를 그래프 구조 안에 포함시킵니다.)
                    2. 더 효율적인 데이터 전처리를 위해 TFRecord 형식으로 데이터를 저장하고  tf.data.TFRecordDataset 를 사용할 수 있습니다.
                * tf.train.Saver().export_meta_graph API 와 tf.train.import_meta_graph API를 사용하여 Training, Test 코드를 각각 실행합니다.(그래프파일인 meta파일을 저장하여 Test시 불러옵니다.)
                * 256x256 크기 이상의 서로 다른 크기의 이미지들을 동시에 학습 할 수 있습니다.
                    * 256x256 크기의 이미지로 학습한 생성 네트워크에 512x512 크기의 이미지를 입력으로 넣어 성능을 평가하기 위한 기능입니다. 다양한 크기의 이미지를 동시에 학습하는 것도 가능합니다. ( 관련 내용 : [Image To Image Translation With Conditional Adversarial Networks Using edges2shoes Dataset 논문의 7p 참고](https://arxiv.org/pdf/1611.07004.pdf))

            * [***Semantic Segmentation***](https://github.com/JONGGON/Tensorflow_Advanced_Tutorials/tree/master/tensorflow_Application/tensorflow_SemanticSegmentation)
                * 자질구레한 문제가 생겨서 코드를 내립니다.
                * 필요한 것? 입력 이미지(.bmp), 정답 이미지들(.npy) 입니다.
                * 정답 이미지들(.npy)은 각각의 1채널 Segmentation map을 channel(depth) 방향으로 쌓은 형태 입니다.(여기서 channel or depth의 수는 segmentation 하고자 하는 class number와 같습니다.)
                    ```
                    예를 들어, (Height, Width, class number) 형태가 됩니다.
                    ```
                * tfrecord를 사용해 데이터를 처리하고 불러옵니다.
                * pixel wise softmax cross entropy, soft dice loss를 지원합니다. 
                * UNET만 구현되어 있습니다.
                    * 다른 네트워크를 구현하고 싶다면 core/model/network에 구현을 하시면 됩니다.
                    
    * ### **Reinforcement Learning**

        * 저는 강화 학습 초보 입니다. 논문을 정독하면서 공부하고 구현하는 것이기 때문에, 아래의 
        주제들을 이해하고 구현하는데 상당히 시간이 많이 걸릴듯 합니다.  

        * [***Policy Gradient with CartPole***](https://github.com/JONGGON/Tensorflow_Advanced_Tutorials/tree/master/tensorflow_ReinForcementLearning/PG_CartPole) 
            * 카트 위에 놓인 막대가 넘어지지 않도록 왼쪽 또는 오른쪽으로 가속시키는 2D 시뮬레이션입니다.
            * 일단 에피소드를 몇 번 진행해보고 이를 평균내어 학습합니다.
            * 몬테카를로 정책 그라디언트(Monte Carlo Policy Gradient) 방법입니다. 
                    
        * [***Deep Q-learning***](https://github.com/JONGGON/Tensorflow_Advanced_Tutorials/tree/master/tensorflow_ReinForcementLearning/DQN)
            * Deep Q-네트워크(Deep Q-networks)를 사용한 방법입니다.
            * 다양한 gym 환경들에 대해서 학습이 가능합니다.(참고 : https://gym.openai.com/envs/#atari) 

        * [***Double Deep Q-learning***](https://github.com/JONGGON/Tensorflow_Advanced_Tutorials/tree/master/tensorflow_ReinForcementLearning/Double_DQN)
            * Double Deep Q-네트워크(Deep Q-networks)를 사용한 방법입니다.
            * Double DQN?             
                
                * DQN이 각 상태에서 잠재적인 액션의 Q 값을 종종 overestimation 한다는 점에서 착안한 방법론 입니다. DDQN을 사용하면, action 선택과 타깃 Q값 생성을 분리함으로써 estimation 값이 크게 나오는 것을 줄일 수 있고, 더 빠르고 안정적으로 학습을 진행할 수 있습니다.
                
                * 구현 방법? 학습 단계에서 target Q 값을 계산할 때 Q 값들에서 최댓값을 구하는 대신 online network에서 가장 큰 Q값을 가지는 action을 선택하고, 해당 action에 대한 target Q값을 target 네트워크에서 생성합니다.

            * 다양한 gym 환경들에 대해서 학습이 가능합니다.(참고 : https://gym.openai.com/envs/#atari) 
        
        * [***Dueling Deep Q-learning***](https://github.com/JONGGON/Tensorflow_Advanced_Tutorials/tree/master/tensorflow_ReinForcementLearning/Dueling_Double_DQN)
            * Dueiling Deep Q-네트워크(Deep Q-networks)를 사용한 방법입니다.
            * Dueling DQN?             
                * Advantage 함수와 Value 함수를 분리하여 계산하고 마지막 계층에서만 조합하여
                조합하여 하나의 Q값으로 만들어 주는 네트워크 입니다.
                * 자세한 내용은 논문의 Introduction을 보시면 됩니다.
                * 본 논문에서는 Double DQN을 기본으로 사용합니다.(선택 가능)  

            * 다양한 gym 환경들에 대해서 학습이 가능합니다.(참고 : https://gym.openai.com/envs/#atari) 
 
>## **개발 환경**
* os : ```window 10.1 64bit, ubuntu 18.04``` 
* python version(`3.6.4`) : `anaconda3 4.4.10` 
* IDE : `pycharm Community Edition 2018.1.2`
    
>## **코드 실행에 필요한 파이썬 모듈** 
* Tensorflow-1.12
* numpy, collections, pandas, random
* matplotlib, scikit-learn, opencv-python, scipy, copy
* tqdm, os, glob, shutil, urllib, zipfile, tarfile

>## **Author**
Kim Jong Gon

medical18@naver.com
