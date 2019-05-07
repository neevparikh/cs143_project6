from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import os

ldir = "results/medium_fewer_bugs/test_latest/images/"
idir = "data/train_img/"

tldir = os.getcwd() + "/data/test_label/"
ldir = os.getcwd() + "/data/train_label/"
idir = os.getcwd() + "/data/train_img/"
tidir = os.getcwd() + "/data/test_img/"
nndir = os.getcwd() + "/data/test_label_nonorm/"

for i in range(0, 8001, 30):
    print(i)
    nnpath = nndir + "label_{}.png".format(i)
    tlpath = tldir + "label_{}.png".format(i)
    lpath = ldir + "label_{}.png".format(i)
    ipath = tidir + "img_{}.png".format(i)
    tipath = tidir + "img_{}.png".format(i)
    if os.path.exists(nnpath) and os.path.exists(tlpath) and os.path.exists(tipath):
        nn_show = np.array(Image.open(nndir + "label_{}.png".format(i))) 
        tlb_show = np.array(Image.open(tldir + "label_{}.png".format(i))) 
        # tlb_show = np.array(Image.open(tldir + "label_{}.png".format(i))) 
        # im_show = np.array(Image.open(idir + "img_{}.png".format(i)))
        tim_show = np.array(Image.open(tidir + "img_{}.png".format(i)))
        fig, axs = plt.subplots(1,3)
        axs[0].imshow(tlb_show * 100)
        # axs[1].imshow(im_show)
        axs[1].imshow(tim_show)
        axs[2].imshow(nn_show * 100)
        # axs[3].imshow(tim_show)
        plt.show()

