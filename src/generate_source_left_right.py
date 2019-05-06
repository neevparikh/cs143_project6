import numpy as np
import os
from utils import save_dir

def main():
    source_left = []
    source_right = []

    source_poses = np.load(os.path.join(
        save_dir, "source_poses.npy"), allow_pickle=True)
    for pose in source_poses:
        source_left.append(pose[13, 1])
        source_right.append(pose[10, 1])

    np.save(os.path.join(save_dir, "source_left.npy"), source_left)
    np.save(os.path.join(save_dir, "source_right.npy"), source_right)

if __name__ == "__main__":
    main()
