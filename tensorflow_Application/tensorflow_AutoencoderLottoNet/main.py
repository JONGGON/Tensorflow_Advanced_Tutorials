from model import model

# optimizers_ selection = "Adam" or "RMSP" or "SGD"
model(TEST=False, optimizer_selection="Adam", learning_rate=0.0009, training_epochs=100000, batch_size=256,
      display_step=10,
      # 전 회차 당첨번호 6자리 입력
      # 반드시 이차원 배열로 선언
      previous_first_prize_number=[[2, 21, 28, 38, 42, 45]], number_of_prediction=5,  regularization = 'L2', scale=0.0001)
