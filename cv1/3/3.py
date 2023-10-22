import numpy as np
import matplotlib.pyplot as plt
import imageio
import os


def gaussian_kernel(size, sigma):
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-(
        (x-(size-1)/2)**2 + (y-(size-1)/2)**2)/(2*sigma**2)), (size, size))
    return kernel / np.sum(kernel)


def gaussian_blur(image, kernel):
    image_row, image_col = image.shape
    kernel_size = kernel.shape[0]
    padding = kernel_size // 2
    blurred_image = np.zeros_like(image)

    padded_image = np.pad(
        image, ((padding, padding), (padding, padding)), mode='constant')
    for i in range(image_row):
        for j in range(image_col):
            image_block = padded_image[i:i+kernel_size, j:j+kernel_size]
            weighted_sum = np.sum(np.multiply(image_block, kernel))
            blurred_image[i, j] = weighted_sum

    return blurred_image


def show(image, size, sigmas):
    kernels = []
    blurred_images = []
    gray_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    for i in range(3):
        kernels.append(gaussian_kernel(size[i], sigmas[i]))
        blurred_images.append(gaussian_blur(gray_image, kernels[i]))
    fig, axes = plt.subplots(2, 2)
    index = -1
    for d in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        x, y = d[0], d[1]
        if index == -1:
            axes[x, y].imshow(image)
            axes[x, y].set_title('Original Image')
        else:
            axes[x, y].imshow(blurred_images[index], cmap='gray')
            axes[x, y].set_title(f"Blurred Image (σ={sigmas[index]})")
            imageio.imwrite(os.path.join(os.path.dirname(
                os.path.abspath(__file__)), f'σ={sigmas[index]}.png'), blurred_images[index])
        axes[x, y].axis('off')
        index += 1
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    image_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'img.png')
    image = imageio.imread(image_path)
    show(image, [3, 5, 7], [1, 3, 5])
