from matplotlib import pyplot as plt
import numpy as np 
from PIL import Image
import os


ldir = "/home/neev/Pictures/data/train_label/"
idir = "/home/neev/Pictures/data/train_img/"

files_label = os.listdir(os.fsencode(ldir))
files_img = os.listdir(os.fsencode(idir))

for i in range(8000, 9000):
    if os.path.exists(ldir + "label_{}.png".format(i)): 
        lb_show = np.array(Image.open(ldir + "label_{}.png".format(i))) * 10
        im_show = np.array(Image.open(idir + "img_{}.png".format(i)))
        fig, axs = plt.subplots(1,2)
        axs[0].imshow(lb_show)
        axs[1].imshow(im_show)
        plt.show()

