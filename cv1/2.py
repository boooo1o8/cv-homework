import numpy as np
import matplotlib.pyplot as plt


def gaussian_kernel(size, sigma):
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-(
        (x-(size-1)/2)**2 + (y-(size-1)/2)**2)/(2*sigma**2)), (size, size))
    return kernel / np.sum(kernel)


def plot_kernel(kernels, sigmas):
    fig, axes = plt.subplots(1, len(kernels))
    for i, (kernel, sigma) in enumerate(zip(kernels, sigmas)):
        im = axes[i].imshow(kernel, cmap='hot', interpolation='nearest')
        axes[i].set_title(f"Sigma = {sigma}")
        fig.colorbar(im, ax=axes[i])
    plt.show()


def show(n, sigmas):
    kernel = []
    for i in range(4):
        kernel.append(gaussian_kernel(n, sigmas[i]))
    plot_kernel(kernel, sigmas)


if __name__ == '__main__':
    show(9, [0.5, 1, 2, 3])
