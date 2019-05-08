from utils import get_body, get_ordered_files, chunk
from generate_data import add_base_args, make_get_path, save_pose
import cv2
import argparse
import os
import numpy as np
from multiprocessing import Pool, current_process


def get_poses(paths):
    body_estimation = get_body()
    poses = []
    subsets = []
    id = current_process()
    print(paths)

    for path in paths:
        print("on path:", path, "id:", id, flush=True)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        pose, subset = body_estimation(img)
        poses.append(pose)
        subsets.append(subset)

    return np.array(poses), np.array(subsets)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("image_dir", help="path to directory with images")
    parser.add_argument("save_prefix", help="start of saved file names")
    parser.add_argument(
        "--save_dir", help="directory to save outputs", default="pose_data")
    parser.add_argument("--threads", help="number of threads to use", type=int,
                        default=8)

    args = parser.parse_args()

    pool = Pool(processes=args.threads)

    def to_full_path(file_name): return os.path.join(args.image_dir, file_name)

    poses_and_subsets = pool.map(
        get_poses,
        chunk(
            list(map(to_full_path, get_ordered_files(args.image_dir))),
            args.threads
        )
    )

    print("after pool")

    full_poses = []
    full_subsets = []

    for poses, subsets in poses_and_subsets:
        full_poses.extend(poses)
        full_subsets.extend(subsets)

    to_iter = (("poses", full_poses), ("subsets", full_subsets))

    for name, values in to_iter:
        np.save(os.path.join(args.save_dir, args.save_prefix + name), values)

if __name__ == "__main__":
    main()
