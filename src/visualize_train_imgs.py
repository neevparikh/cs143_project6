from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import os


ldir = os.getcwd() + "/data/test_label/"
idir = os.getcwd() + "/data/train_img/"
tidir = os.getcwd() + "/data/test_img/"

for i in range(400, 401):
    i = 400
    path = ldir + "label_{}.png".format(i)
    if os.path.exists(path):
        lb_show = np.array(Image.open(ldir + "label_{}.png".format(i))) 
        im_show = np.array(Image.open(idir + "img_{}.png".format(i)))
        tim_show = np.array(Image.open(tidir + "img_{}.png".format(i)))
        fig, axs = plt.subplots(1,3)
        axs[0].imshow(lb_show * 10)
        axs[1].imshow(im_show)
        axs[2].imshow(tim_show)
        plt.show()

