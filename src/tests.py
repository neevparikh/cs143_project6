import sys
import cv2
sys.path.insert(0, "pytorch-openpose")
sys.path.insert(0, "gan")
import argparse
import pickle
from gan_wrapper import GANWrapper
from pose import get_pose_normed_estimate, get_pose_estimate 

# Creates command line parser
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="path to source video")
    parser.add_argument("target", help="path to target video")
    parser.add_argument("--mode", choices=["train", "transfer"], required=True,
                        help="training mode or transfer mode")
    args = parser.parse_args()

    source = args.source
    target = args.target
    mode = args.mode

    if mode == "train":
        target_poses = get_pose_estimate(target)
        print(target_poses)
    else:
        norm_source_poses = get_pose_normed_estimate(source, target)[0]
        print(norm_source_poses)

if __name__ == "__main__":
    main()
