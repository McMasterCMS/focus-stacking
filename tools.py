import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def load_image(img_path):
    img_color = Image.open(img_path)
    img_gray = img_color.convert('L')
    return np.asarray(img_gray), np.asarray(img_color)


def show_images(img_list):


    fig = plt.figure(figsize=(14,7))
    num_imgs = len(img_list)

    for i, img in enumerate(img_list):
        plt.subplot(1, num_imgs, i+1)
        if len(img.shape) == 3:
            plt.imshow(img)
        else:
            plt.imshow(img, cmap='gray', vmin=0, vmax=255)
