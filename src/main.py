import sys
import cv2
import argparse
import pickle
from gan_wrapper import GANWrapper
from pose import get_pose_normed_estimate, get_pose_estimate

# Creates command line parser
def main():
    parser = argparse.ArgumentParser()

    #TODO
    parser.add_argument("source", help="path to source video")
    parser.add_argument("target", help="path to target video")
    parser.add_argument("--mode", choices=["train", "transfer"], required=True, help="training mode or transfer mode")
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
        target_poses, target_subsets = get_pose_estimate(target, regen=regen)
        gan_model = GANWrapper(source, target, mode)
        gan_model.train(target_subsets)
    else:
        norm_source_poses, source_poses, source_subsets, target_poses, target_subsets = get_pose_normed_estimate(source, target, regen=regen)
        #TODO
        try:
            gan_model = pickle.load(open("outputs/gan/trained_gan.pth", "rb"))
        except FileNotFoundError:
            print("GAN model not found. Please retrain the GAN.")
            sys.exit()
        transfered_video = gan_model.apply_mapping(norm_source_poses, source_subsets)
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        video = cv2.VideoWriter('video/pose_normed.mp4', fourcc, 30,(720, 480))
        for frame in transfered_video:
            video.write(frame)

if __name__ == "__main__":
    main()
