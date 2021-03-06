import argparse

from generate_data import add_base_args, make_get_path, save_pose
from utils import PoseNormalizer, save_dir
import numpy as np
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual_scale", type=float, default=None,
                        help="manual scale factor")
    parser.add_argument("--manual_translate", type=float, default=None,
                        help="manual translation factor")
    add_base_args(parser)

    args = parser.parse_args()

    target_poses = np.load(os.path.join(
        save_dir, "train_poses.npy"), allow_pickle=True)
    target_subsets = np.load(os.path.join(
        save_dir, "train_subsets.npy"), allow_pickle=True)
    source_poses = np.load(os.path.join(
        save_dir, "test_poses.npy"), allow_pickle=True)
    source_subsets = np.load(os.path.join(
        save_dir, "test_subsets.npy"), allow_pickle=True)

    assert len(source_poses) == len(source_subsets)
    assert len(target_poses) == len(target_subsets)

    source_left, source_right, target_left, target_right = [], [], [], []

    if args.manual_translate is None or args.manual_scale is None:
        iter = (
            ("source", source_left, source_right, source_poses, source_subsets),
            ("target", target_left, target_right, target_poses, target_subsets)
        )

        for name, left, right, poses, subsets in iter:
            for i in range(poses.shape[0]):
                l_ankle = poses[i][np.where(poses[i][:, 3] == 13)].shape[0] != 0
                r_ankle = poses[i][np.where(poses[i][:, 3] == 10)].shape[0] != 0

                # Check if either right or left ankle is missing
                if l_ankle and r_ankle and subsets[i][:, 19].size != 0 and \
                        np.min(subsets[i][:, 19]) >= 18:
                    left.append(poses[i][13, 1])
                    right.append(poses[i][10, 1])
                    print(name, 'frame kept:', i, flush=True)
                else:
                    print(name, 'ankle dropped:', i, flush=True)

        assert len(source_right) == len(source_left)
        assert len(target_right) == len(target_left)

        np.save(os.path.join(save_dir, "source_left.npy"), source_left)
        np.save(os.path.join(save_dir, "source_right.npy"), source_right)

        np.save(os.path.join(save_dir, "target_left.npy"), target_left)
        np.save(os.path.join(save_dir, "target_right.npy"), target_right)

        source_dict = {
            "left": np.array(source_left),
            "right": np.array(source_right)
        }

        target_dict = {
            "left": np.array(target_left),
            "right": np.array(target_right)
        }
    else:
        source_dict = None
        target_dict = None


    pose_normalizer = PoseNormalizer(source_dict, target_dict, epsilon=5,
                                     alpha=1, manual_scale=args.manual_scale,
                                     manual_translate=args.manual_translate)

    norm_source = source_poses.copy()
    transformed_all = pose_normalizer.transform_pose_global(
        norm_source
    )

    np.save(os.path.join(save_dir, "normed_source_pose.npy"),
            np.array(transformed_all))

    test_path_label = make_get_path(False, True)

    for i, (pose, subsets) in enumerate(zip(transformed_all, source_subsets)):
        if i >= args.max_frames:
            break
        save_pose(pose, subsets, test_path_label(i), args.height, args.width)
        print('test written', i, flush=True)


if __name__ == "__main__":
    main()
