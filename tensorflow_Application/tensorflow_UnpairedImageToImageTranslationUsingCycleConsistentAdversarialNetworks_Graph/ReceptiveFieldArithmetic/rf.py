'''
Image-to-Image Translation with Conditional Adversarial Networks 논문에서 등장하는 PatchGan이 실제로 70의 ReceptiveField 크기를 갖는지 확인 해보자.
'''

def ReceptiveFieldSizeCalculator(input_size=5, weight_size=3, stride=2, padding=1,
                                               input_start_position=0.5, input_rf_size=1,input_j=1):
    '''
    알아야 되는 식
    1. convolution 식 -> (input_size - weight_size + 2*padding)/stride + 1
    2. 두개의 인접한 weigth 간의 거리를 구하는 식 -> output_j = input_j*stride
    3. Receptive field 크기를 구하는 식 -> output_rf_size = input_rf_size + (weight_size-1)*input_j
    4. 왼쪽 weight의 중심좌표를 구하는 식 -> output_start_position = input_start_position + ((weight_size-1)/2 - padding)*input_j
    '''
    output_size = (input_size - weight_size + 2 * padding) / stride + 1
    output_j = input_j*stride
    output_rf_size = input_rf_size + (weight_size-1)*input_j
    output_start_position = input_start_position + ((weight_size-1)/2 - padding)*input_j

    return output_size , output_j, output_rf_size, output_start_position

if __name__ == "__main__":
    '''
    input_start_position 의 의미? 왼쪽 weight의 중심좌표
    input_rf_size 의 의미?  현재의 receptive field의 크기 , 처음엔 1
    input_j 는 distance of two adjacent feature를 의미한다 즉, 두개의 인접한 weigth간의 거리, 처음엔 1 
    '''

    input_size , input_j, input_rf_size, input_start_position = ReceptiveFieldSizeCalculator(input_size=256, weight_size=4, stride=2, padding=1, # 콘볼루션 식을 위함
                                               input_start_position=0.5, input_rf_size=1, input_j=1) # receptive field size 계산을 위함
    print("ReceptiveField 크기 : {}".format(input_rf_size))
    #input_size -> 128
    input_size , input_j, input_rf_size, input_start_position = ReceptiveFieldSizeCalculator(input_size=input_size, weight_size=4, stride=2, padding=1, # 콘볼루션 식을 위함
                                               input_start_position=input_start_position, input_rf_size=input_rf_size, input_j=input_j) # receptive field size 계산을 위함
    print("ReceptiveField 크기 : {}".format(input_rf_size))
    #input_size -> 64
    input_size , input_j, input_rf_size, input_start_position = ReceptiveFieldSizeCalculator(input_size=input_size, weight_size=4, stride=2, padding=1, # 콘볼루션 식을 위함
                                               input_start_position=input_start_position, input_rf_size=input_rf_size, input_j=input_j) # receptive field size 계산을 위함
    print("ReceptiveField 크기 : {}".format(input_rf_size))
    #input_size -> 32
    input_size , input_j, input_rf_size, input_start_position = ReceptiveFieldSizeCalculator(input_size=input_size, weight_size=4, stride=1, padding=1, # 콘볼루션 식을 위함
                                               input_start_position=input_start_position, input_rf_size=input_rf_size, input_j=input_j) # receptive field size 계산을 위함
    print("ReceptiveField 크기 : {}".format(input_rf_size))
    #input_size -> 31
    input_size , input_j, input_rf_size, input_start_position = ReceptiveFieldSizeCalculator(input_size=input_size, weight_size=4, stride=1, padding=1, # 콘볼루션 식을 위함
                                               input_start_position=input_start_position, input_rf_size=input_rf_size, input_j=input_j) # receptive field size 계산을 위함
    print("ReceptiveField 크기 : {}".format(input_rf_size))
