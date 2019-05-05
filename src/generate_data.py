import os
import sys
import cv2
import argparse
import numpy as np
from utils import (draw_bodypose, get_body, get_pose_normed_estimate,
                   transform_frame, loop_frame)

def main():
    # Creates command line parser
    parser = argparse.ArgumentParser()

    parser.add_argument("target", help="path to target video")
    parser.add_argument("source", help="path to source video")
    parser.add_argument("--height", type=int, default=1024,
                        help="height for saved video")
    parser.add_argument("--width", type=int, default=512,
                        help="width for saved video")
    parser.add_argument("--rotated", choices=["true", "false"], required=True)
    parser.add_argument("--phase", choices=["train", "test"], required=True)
    parser.add_argument("--regen", choices=["true", "false"], default='true',
                        required=True)
    parser.add_argument("--max_frames", type=float, default=float('inf'))

    args = parser.parse_args()
    target = args.target
    rotated = args.rotated
    phase = args.phase
    regen = args.regen
    height = args.height
    width = args.width
    source = args.source
    max_frames = args.max_frames

    data = get_pose_normed_estimate(source, target, regen=regen,
                                    rotate=rotated, max_frames=max_frames,
                                    height=height, width=width)

    normed_source = data["normed_source"]
    source_subsets = data["source_subsets"]
    target_poses = data["target_poses"]
    target_subsets = data["target_subsets"]
    source_indexes = data["source_indexes"]
    target_indexes = data["target_indexes"]

    def make_get_path(is_train, is_label):
        type_name = 'label' if is_label else 'img'
        path_b = "data/{}_{}/{}_".format(('train' if is_train else 'test'),
                                         type_name, type_name)

        def get_path(counter, path_b=path_b):
            return "{}{}.png".format(path_b, counter)

        return get_path

    def save_pose(pose, subset, path):
        canvas = np.ones((args.height, args.width, 3), dtype='uint8')
        target_image = draw_bodypose(canvas, pose, subset)
        cv2.imwrite(path, target_image)

    if phase =='train':

        assert len(target_poses) == len(target_subsets)
        assert len(target_frames_indexes) == len(target_subsets)

        target_counter_index = 0

        train_path_label = make_get_path(True, True)
        train_path_img = make_get_path(True, False)

        def loop_target(frame, counter):
            if counter == target_indexes[target_counter_index]:
                cv2.imwrite(train_path_img(counter), frame)

                save_pose(target_poses[counter], target_subsets[counter],
                          train_path_label(counter))

                print("train writing:", counter)

                target_counter_index += 1

        loop_frame(target, loop_target)

    else:
        test_path_img = make_get_path(False, True)

        for pose, subsets, index in zip(normed_source, source_subsets,
                                        source_indexes):
            save_pose(pose, subsets, test_path_label(index))

            print('test', i, flush=True)

if __name__ == "__main__":
    main()
