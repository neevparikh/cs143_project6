import sys
import cv2
sys.path.insert(0, "pytorch-openpose")
sys.path.insert(0, "gan")
import argparse
import pickle
from gan_wrapper import GANWrapper
from pose import get_pose_normed_estimate, get_pose_estimate 
from matplotlib import pyplot as plt

# Creates command line parser
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="path to source video")
    parser.add_argument("target", help="path to target video")
    parser.add_argument("--mode", choices=["train", "transfer", "create_dir"], required=True,
                        help="training mode or transfer mode")
    parser.add_argument("--regen", action="store_true", dest="regen",
                        help="regenerate the pickles")
    parser.add_argument("--no-regen", action="store_false", dest="regen",
                        help="do not regenerate the pickles")
    args = parser.parse_args()

    source = args.source
    target = args.target
    mode = args.mode
    regen = args.regen

    if mode == "train":
        target_poses, target_subsets = get_pose_estimate(target, regen=regen, rotate=False)
        gan = GANWrapper(source, target, mode)
    elif mode == "create_dir":
        target_poses, target_subsets = get_pose_estimate(target, regen=regen, rotate=False)
        source_poses, source_subsets = get_pose_estimate(source, regen=regen, rotate=False)
        gan = GANWrapper(source, target, mode)
        gan.create_training_data(source, source_poses, source_subsets, "data/source/")
        gan.create_training_data(target, target_poses, target_subsets, "data/target/")
    else:
        norm_source_poses = get_pose_normed_estimate(source, target, regen=regen)[0]

if __name__ == "__main__":
    main()
