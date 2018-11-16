import VariationalAutoencoder as VA

# optimizers_ selection = "Adam" or "RMSP" or "SGD"
# latent_number는 2의 배수인 양수여야 한다.
'''
targeting = False 일 때는 숫자를 무작위로 생성하는 VAE 생성 - General VAE
targeting = True 일 때는 숫자를 타게팅 하여 생성하는 VAE 생성 - Conditional VAE
'''
# batch normalization은 Hidden Layer에만 추가합니다. 또한 활성화 함수전에 적용합니다.
# regularization -> batch_norm = False 일때, L2 or L1 or nothing
VA.model(TEST=True, targeting=False, latent_number=32, optimizer_selection="Adam", \
         learning_rate=0.001, training_epochs=1, batch_size=512, display_step=1, batch_norm=True, regularization='L2', scale=0.0001)
