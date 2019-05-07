from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import os

ldir = "results/medium_fewer_bugs/test_latest/images/"
idir = "data/train_img/"

for i in range(1000, 9100):
    path = ldir + "label_{}_input_label.jpg".format(i)
    if os.path.exists(path):
        lb_show = np.array(Image.open(ldir + "label_{}_synthesized_image.jpg".format(i)))
        im_show = np.array(Image.open("/home/ryan/Pictures/train_demo/epoch003_synthesized_image.jpg"))
        fig, axs = plt.subplots(1,2)
        axs[0].imshow(lb_show)
