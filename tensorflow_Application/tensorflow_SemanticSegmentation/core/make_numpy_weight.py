import os
import numpy as np
import tensorflow as tf

# 가중치 저장 - 약간 생소한 API지만 유용.

''' <<< 종곤-대경 협약 >>>

LearnedInfo 폴더 아래 
 minmax.npy 파일
 /Weight/weight, bias 파일
 /BatchNorm/ 나머지 파일

'''
def make_numpy_weight(model_path, folder_path, input_range, WeightSelection=None):

    weight_save_path = os.path.join(folder_path, "Weight")
    norm_save_path = os.path.join(folder_path, "BatchNorm")

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if not os.path.exists(weight_save_path):
        os.makedirs(weight_save_path)
    if not os.path.exists(norm_save_path):
        os.makedirs(norm_save_path)

    input_range = np.array([input_range])
    np.save(os.path.join(folder_path, "input_range.npy"), input_range)

    # 1, checkpoint 읽어오는 것
    reader = tf.train.NewCheckpointReader(model_path)
    dtype = list(reader.get_variable_to_dtype_map().values())[0]

    ''' 2. tf.train.NewCheckpointReader에도
    reader.get_variable_to_dtype_map() -> 이름 , dype 반환 or reader.get_variable_to_shape_map() 이름 , 형태 반환 
    하는데 사전형이라 순서가 중구난방이다.
    요 아래의 것은 리스트 형태로 name, shape이 순서대로 나온다.
    '''
    name_shape = tf.contrib.framework.list_variables(model_path)
    with open(os.path.join(folder_path, "name_shape_info.txt"), mode='w') as f:
        f.write("                      < weight 정보 >\n\n")
        f.write("파일 개수 : {}개\n\n".format(len(name_shape)))
        f.write("------------------- 1. data type ---------------------\n\n")
        f.write("{} \n\n".format(str(dtype).strip("<>").replace(":", " :")))
        print("-----------------------------------------------------------")
        print("<<< 총 파일 개수 : {} >>>\n".format(len(name_shape)))

        f.write("<<< {} checkpoint 로 만듦 >>>\n\n".format(WeightSelection))
        f.write("<<< input range : {} >>>\n\n".format(input_range))

        f.write("-------------- 2. weight name, shape ----------------\n\n")
        for name, shape in name_shape:
            # 앞의 {0 : variable_scope}/{1 : network} 이름 빼버리기
            seperated = name.split("/")[2:]
            joined = "_".join(seperated)
            shape = str(shape).strip('[]')
            print("##################################################")
            print("<<< weight : {}.npy >>>".format(joined))
            print("<<< shape : ({}) >>>".format(shape))
            f.write("<<< {}.npy >>>\n<<< shape : ({}) >>>\n\n".format(joined, shape))

            # weight npy로 저장하기
            # 1. Batchnorm
            if joined.split("_")[-1] == "gamma" or joined.split("_")[-1] == "beta":
                np.save(os.path.join(norm_save_path, joined), reader.get_tensor(name))
            # 2. Weight
            else:
                np.save(os.path.join(weight_save_path, joined), reader.get_tensor(name))