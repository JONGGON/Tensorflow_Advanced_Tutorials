import matplotlib.pyplot as plt
from sklearn import decomposition
from tensorflow.examples.tutorials.mnist import input_data


def PCA(n_components=2, show_reconstruction_image=True):
    # 1. PCA수행
    mnist = input_data.read_data_sets("")
    pca = decomposition.PCA(n_components=n_components)
    pca.fit(mnist.train.images)
    pca_codes = pca.transform(mnist.test.images)

    # 2. 재건
    if show_reconstruction_image:
        row_size = 10
        column_size = 10
        pca_reconstruction = pca.inverse_transform(pca_codes[:row_size * column_size])
        fig_r, ax_r = plt.subplots(row_size, column_size, figsize=(column_size, row_size))
        fig_r.suptitle('PCA Reconstruction')
        for j in range(row_size):
            for i in range(column_size):
                ax_r[j][i].grid(False)
                ax_r[j][i].set_axis_off()
                ax_r[j][i].imshow(pca_reconstruction[i + j * column_size].reshape((28, 28)), cmap='gray')
        fig_r.savefig("PCA_Reconstruction.png")
        plt.show()

    return pca_codes


if __name__ == "__main__":
    PCA(n_components=2, show_reconstruction_image=True)
else:
    print("model imported")
