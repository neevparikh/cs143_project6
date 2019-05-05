from utils import get_pose_normed_estimate, loop_frame
from generate_data import add_base_args, make_get_path, save_pose
import cv2
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="path to target video")
    parser.add_argument("--regen", choices=["true", "false"], default='true')
    add_base_args(parser)

    args = parser.parse_args()

    data = get_pose_normed_estimate(None, args.target, regen_target=args.regen,
                                    rotate=args.rotated,
                                    max_frames=args.max_frames,
                                    height=args.height, width=args.width)


    target_poses = data["target_poses"]
    target_subsets = data["target_subsets"]
    target_indexes = data["target_indexes"]

    assert len(target_poses) == len(target_subsets)
    assert len(target_indexes) == len(target_subsets)

    target_counter_index = 0

    train_path_label = make_get_path(True, True)
    train_path_img = make_get_path(True, False)

    def loop_target(frame, counter):
        if counter == target_indexes[target_counter_index]:
            cv2.imwrite(train_path_img(counter), frame)

            save_pose(args.height, args.width, target_poses[counter],
                      target_subsets[counter], train_path_label(counter))

            print("train writing:", counter)

            target_counter_index += 1

    loop_frame(args.target, args.max_frames, loop_target)

if __name__ == "__main__":
    main()
