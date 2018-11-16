from FNN import model

# optimizers_ selection = "Adam" or "RMSP" or "SGD"
# batch normalization은 Hidden Layer에만 추가합니다. 또한 활성화 함수전에 적용합니다.
# regularization -> batch_norm = False 일때, L2 or L1 or nothing
model(TEST=True, optimizer_selection="Adam", learning_rate=0.001, training_epochs=50,
batch_size = 256, display_step = 1, batch_norm = True, regularization = 'L2', scale = 0.0001)
