import argparse

import cv2

from generate_data import add_base_args, make_get_path, save_pose
from utils import get_pose_normed_estimate, loop_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="path to target video")
    parser.add_argument("--regen", action='store_true', dest='regen') 
    parser.add_argument("--no-regen", action='store_false', dest='regen') 
    add_base_args(parser)

    args = parser.parse_args()

    data = get_pose_normed_estimate(None, args.target,
                                    regen_source=None,
                                    regen_target=args.regen,
                                    regen_norm=None,
                                    rotated=args.rotated,
                                    height=args.height, width=args.width,
                                    max_frames=args.max_frames)

    target_poses = data["target_poses"]
    target_subsets = data["target_subsets"]
    target_indexes = data["target_indexes"]

    assert len(target_poses) == len(target_subsets)
    assert len(target_indexes) == len(target_subsets)

    target_counter_index = 0

    train_path_label = make_get_path(True, True)
    train_path_img = make_get_path(True, False)

    def loop_target(img, counter):
        nonlocal target_counter_index

        if counter == target_indexes[target_counter_index]:
            cv2.imwrite(train_path_img(target_counter_index), img)

            save_pose(target_poses[target_counter_index],
                      target_subsets[target_counter_index],
                      train_path_label(target_counter_index),
                      args.height, args.width)

            print("train writing:", target_counter_index)

            target_counter_index += 1

    loop_frame(args.target, args.max_frames, loop_target, args.rotated,
               args.width, args.height)


if __name__ == "__main__":
    main()
