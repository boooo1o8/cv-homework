import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
import imageio


def gaussian_kernel(size, sigma):
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-(
        (x-(size-1)/2)**2 + (y-(size-1)/2)**2)/(2*sigma**2)), (size, size))
    return kernel / np.sum(kernel)


def show(size, sigmas):
    fig, axes = plt.subplots(1, len(sigmas), figsize=(
        12, 4), subplot_kw={'projection': '3d'})
    pad = size // 2
    xx = np.arange(-pad, -pad + size, 1)
    yy = np.arange(-pad, -pad + size, 1)
    X, Y = np.meshgrid(xx, yy)

    for i, sigma in enumerate(sigmas):
        K = gaussian_kernel(size, sigma)
        Z = K
        ax = axes[i]
        ax.plot_surface(X, Y, Z)
        ax.set_title(f"Sigma = {sigma}")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    show(9, [0.5, 1, 2, 3])
