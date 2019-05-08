import argparse

import cv2

from generate_data import add_base_args, make_get_path, save_pose
from utils import get_pose_normed_estimate, loop_frame, save_dir
from tqdm import tqdm
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    add_base_args(parser)
    args = parser.parse_args()

    poses = np.load("pose_data/train_poses.npy", allow_pickle=True)
    subsets = np.load("pose_data/train_subsets.npy", allow_pickle=True)

    train_path_label = make_get_path(True, True)

    for i in tqdm(range(min(poses.shape[0], args.max_frames))):
        save_pose(poses[i], subsets[i], train_path_label(
            i), args.height, args.width)

if __name__ == "__main__":
    main()
