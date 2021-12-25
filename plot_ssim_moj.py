import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import skimage
import csv
import glob
import cv2
from skimage import io
from skimage.color import rgb2gray
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
list_ = []
path = glob.glob("*.bmp") #or jpg
def read_img(img_list, img):
    #n = cv2.imread(img).astype('float32')
    n=io.imread(img)
    img_list.append(n)
    return img_list

novi = [read_img(list_, img) for img in path]

def usporedba(img,img1):
    original = img
    kopija = img1
    print(original.shape)
    fig = plt.figure(figsize = (8,4))
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(original)
    original_r = original.copy() # Red Channel
    original_r[:,:,1] = original_r[:,:,2] = 0 # set G,B pixels = 0
    original_g = original.copy() # Green Channel
    original_g[:,:,0] = original_r[:,:,2] = 0 # set R,B pixels = 0
    original_b = original.copy() # Blue Channel
    original_b[:,:,0] = original_b[:,:,1] = 0 # set R,G pixels = 0
    plot_image = np.concatenate((original_r, original_g, original_b),   axis=1)
    plt.figure(figsize = (10,4))
    plt.imshow(plot_image)
    plt.close('all')
    original=rgb2gray(original)
    kopija=rgb2gray(kopija)
    img = img_as_float(original)
    #pd.DataFrame(img).to_csv("slika.csv")

    img1 = img_as_float(kopija)
    rows, cols = img.shape
    rows1, cols1 = img1.shape
    noise = np.ones_like(img) * 0.2 * (img.max() - img.min())
    noise[np.random.random(size=noise.shape) > 0.5] *= -1

    def mse(x, y):
      return np.linalg.norm(x - y)

    img_noise = img + noise

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 4),
                            sharex=True, sharey=True)
    ax = axes.ravel()

    mse_none = mse(img, img)
    ssim_none = ssim(img, img, data_range=img.max() - img.min())
    mse_noise = mse(img, img_noise)
    ssim_noise = ssim(img, img_noise,
                    data_range=img_noise.max() - img_noise.min())

    mse_const = mse(img, img1)
    ssim_const = ssim(img, img1,
                    data_range=img1.max() - img1.min())
    print((ssim_const))

    f = open("ssim.txt", 'a')
    f.write(str(ssim_const)+'\n')
    f.close()

    label = 'MSE: {:.2f}, SSIM: {:.2f}'

    ax[0].imshow(img, cmap=plt.cm.gray, vmin=0, vmax=1)
    ax[0].set_xlabel(label.format(mse_none, ssim_none))
    ax[0].set_title('Originalna slika')

    ax[1].imshow(img_noise, cmap=plt.cm.gray, vmin=0, vmax=1)
    ax[1].set_xlabel(label.format(mse_noise, ssim_noise))
    ax[1].set_title('Slika sa šumom')

    ax[2].imshow(img1, cmap=plt.cm.gray, vmin=0, vmax=1)
    ax[2].set_xlabel(label.format(mse_const, ssim_const))
    ax[2].set_title('Testirana slika')

    plt.tight_layout()
    plt.show()


for x in range(len(list_)-1):
    usporedba(list_[x],list_[x+1])
    print(x)