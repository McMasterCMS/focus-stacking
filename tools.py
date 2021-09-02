import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import glob
from scipy.ndimage.filters import gaussian_filter


def load_image(img_path):
    img_color = Image.open(img_path)
    img_gray = img_color.convert('L')
    return np.asarray(img_gray), np.asarray(img_color)


def load_images(img_folder_path):
    img_paths = glob.glob(img_folder_path + "/*.png")

    imgs_color = []
    imgs_gray = []
    for path in img_paths:

        img_color, img_gray = load_image(path)
        imgs_color.append(img_color)
        imgs_gray.append(img_gray)

    return imgs_gray, imgs_color


def show_images(img_list):

    fig = plt.figure(figsize=(10,20))
    num_imgs = len(img_list)

    for i, img in enumerate(img_list):
        plt.subplot(num_imgs, 1, i+1)
        if len(img.shape) == 3:
            plt.imshow(img)
        else:
            plt.imshow(img, cmap='gray', vmin=0, vmax=255)

    plt.tight_layout()


def construct_focused_image(img_grad_list, img_gray_list, img_color_list):

    imgs_grad_blurred = []
    for img_grad in imgs_grad:
        imgs_grad_blurred.append(gaussian_filter(img_grad, sigma=10))

    imgs_gray = np.array(img_gray_list)
    sharpest_indices = np.argmax(np.array(imgs_grad_blurred), axis=0)
    focused_img = np.take_along_axis(imgs_gray, np.expand_dims(sharpest_indices, axis=0), axis=0)

    return focused_img_gray, sharpest_indices
