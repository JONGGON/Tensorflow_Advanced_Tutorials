from model import Word2Vec

# optimizers_ selection = "Adam" or "RMSP" or "SGD"
# weight_selection 은 encoder, decoder 임베디중 어떤것을 사용할 것인지 선택하는 변수
# weight_sharing=True시 weight_selection="decoder"라고 설정해도 encoder의 embedding_matrix 로 강제 설정된다.
Word2Vec(TEST=False, tSNE=True, model_name="Word2Vec", weight_selection="encoder",  # encoder or decoder
         vocabulary_size=30000, tSNE_plot=200, similarity_number=8,
         # similarity_number -> 비슷한 문자 출력 개수
         # num_skip : 하나의 문장당 num_skips 개의 데이터를 생성
         validation_number=30, embedding_size=128, batch_size=128, num_skips=2, window_size=1,
         negative_sampling=64, optimizer_selection="SGD", learning_rate=0.1, training_epochs=1000,
         display_step=1, weight_sharing=False)
