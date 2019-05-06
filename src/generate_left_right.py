import numpy as np
import os
from utils import save_dir

def main():
    for which_type in ["source", "target"]:

        left = []
        right = []

        poses = np.load(os.path.join(
            save_dir, "{}_poses.npy".format(which_type)), allow_pickle=True)
        for pose in poses:
            left.append(pose[13, 1])
            right.append(pose[10, 1])

        np.save(os.path.join(save_dir, "{}_left.npy".format(which_type)), left)
        np.save(os.path.join(save_dir, "{}_right.npy".format(which_type)), right)

if __name__ == "__main__":
    main()
