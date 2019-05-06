from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import os


ldir = os.getcwd() + "/data/test_label/"
idir = os.getcwd() + "/data/train_img/"

for i in range(0, 10):
    path = ldir + "label_{}.png".format(i)
    if os.path.exists(path):
        lb_show = np.array(Image.open(ldir + "label_{}.png".format(i))) 
        im_show = np.array(Image.open(idir + "img_{}.png".format(i)))
        fig, axs = plt.subplots(1,2)
        axs[0].imshow(lb_show * 10)
        axs[1].imshow(im_show)
        plt.show()

