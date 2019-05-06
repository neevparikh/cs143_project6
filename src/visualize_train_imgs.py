from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import os


ldir = "data/test_label/"

for i in range(0, 10):
    if os.path.exists(ldir + "label_{}".format(i)):
        lb_show = np.array(Image.open(ldir + "label_{}.png".format(i)))
        # im_show = np.array(Image.open(ldir + "label_{}_input_label.jpg".format(i)))
        fig, axs = plt.subplots(1,2)
        axs[0].imshow(lb_show)
        # axs[1].imshow(im_show)
        plt.show()

