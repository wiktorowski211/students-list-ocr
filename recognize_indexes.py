import preprocess
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


def show(img):
    plt.figure(figsize=(12, 12))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.show()


def call(image):
    page, _ = preprocess.into_page(image)
    show(page)

    return ["12334", "1241421"]
