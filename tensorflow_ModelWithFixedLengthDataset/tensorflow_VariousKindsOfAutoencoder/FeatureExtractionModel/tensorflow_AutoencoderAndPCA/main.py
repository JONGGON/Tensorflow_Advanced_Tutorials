import Autoencoder

# optimizers_ selection = "Adam" or "RMSP" or "SGD"
# model_name = "Convolution_Autoencoder" or "Autoencoder"
# batch normalization은 Hidden Layer에만 추가합니다. 또한 활성화 함수전에 적용합니다.
# regularization -> batch_norm = False 일때, L2 or L1 or nothing
Autoencoder.model(TEST=False, Comparison_with_PCA=True, optimizer_selection="Adam", model_name = "Autoencoder", learning_rate=0.001,
                  training_epochs=1, batch_size=512,
                  display_step=1, batch_norm=False, regularization=' ', scale=0.0001)
