from utils import get_pose_normed_estimate
from generate_data import add_base_args, make_get_path, save_pose
import cv2
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="path to target video")
    add_base_args(parser)

    args = parser.parse_args()

    get_pose_normed_estimate(args.source, None, True, True, True,
                             rotated=args.rotated, height=args.height,
                             width=args.width, max_frames=args.max_frames,)


if __name__ == "__main__":
    main()