import DenoisingAutoencoder as DA

# optimizers_ selection = "Adam" or "RMSP" or "SGD"
# model_name =  CDA -> ConvolutionDenosingAutoencoder" or DA -> DenosingAutoencoder
# batch normalization은 Hidden Layer에만 추가합니다. 또한 활성화 함수전에 적용합니다.
# regularization -> batch_norm = False 일때, L2 or L1 or nothing
DA.model(TEST=True, Comparison_with_PCA=True, corrupt_probability=0.5,
         optimizer_selection="Adam", model_name="CDA", learning_rate=0.001, training_epochs=1, batch_size=256,
         display_step=1, batch_norm=True, regularization='L1', scale=0.0001)
