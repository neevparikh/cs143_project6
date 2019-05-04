import os
import sys
import cv2
import argparse
import numpy as np
from utils import draw_bodypose, get_body, get_pose_normed_estimate

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

    if os.path.isfile(target):
        video = cv2.VideoCapture(target)
    else:
        raise FileNotFoundError

    data = get_pose_normed_estimate(source, target, regen=regen,
                                    rotate=rotated, max_frames=max_frames,
                                    height=height, width=width)

    normed_source = data["normed_source"]
    source_subsets = data["source_sub"]
    target_poses = data["target_pose"]
    target_subsets = data["target_sub"]
    source_frames = data["source_frame"]
    target_frames = data["target_frame"]

    for i in range(len(source_frames)):
        canvas = np.ones((args.height, args.width, 3), dtype='uint8')
        if phase =='train':

            target_image = draw_bodypose(canvas, target_poses[i],
                                         target_subsets[i])
            target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

            cv2.imwrite('data/train_label/target_pose_' + str(i) + '.png',
                        target_image)
            cv2.imwrite('data/train_img/target_frame_' + str(i) + '.png',
                        target_frames[i])

        elif phase == 'test':

            source_image = draw_bodypose(canvas, normed_source[i],
                                         source_subsets[i])
            source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)

            cv2.imwrite('data/test_label/source_pose_' + str(i) + '.png',
                        source_image)
            cv2.imwrite('data/test_img/source_frame_' + str(i) + '.png',
                        source_frames[i])

        print('Written', i, flush=True)
    video.release()

if __name__ == "__main__":
    main()
