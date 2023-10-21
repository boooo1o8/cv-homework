import numpy as np
import matplotlib.pyplot as plt
import imageio
import os


def zero_padding(image_array, padding_size):
    padded_image = np.pad(image_array, ((padding_size, padding_size),
                          (padding_size, padding_size), (0, 0)), mode='constant')
    return padded_image


def replicate_padding(image_array, padding_size):
    padded_image = image_array.copy()
    height, width, channels = image_array.shape

    for i in range(padding_size):
        padded_image = np.insert(
            padded_image, 0, padded_image[0], axis=0)  # 复制第一行
        padded_image = np.insert(
            padded_image, height + 1, padded_image[height], axis=0)  # 复制最后一行
        padded_image = np.insert(
            padded_image, 0, padded_image[:, 0], axis=1)  # 复制第一列
        padded_image = np.insert(
            padded_image, width + 1, padded_image[:, width], axis=1)  # 复制最后一列

    return padded_image


def show(image, padding_size):
    image_array = np.array(image)
    padded_image_zero = zero_padding(image_array, padding_size)
    padded_image_replicate = replicate_padding(image_array, padding_size)
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(image_array)
    axs[0].set_title('Original Image')
    axs[1].imshow(padded_image_zero)
    axs[1].set_title('Zero Padding')
    axs[2].imshow(padded_image_replicate)
    axs[2].set_title('Replicate Padding')
    plt.show()


if __name__ == '__main__':
    image_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'img2.jpg')
    image = imageio.imread(image_path)
    show(image, 50)
