#!/usr/bin/env python3

import cv2
import os
import argparse
import re


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_folder", help="path to image folder")
    parser.add_argument("video_name", 
            help="name of output video with file extension")
    parser.add_argument("--pattern", help="regex pattern to use", default=None)
    parser.add_argument("--multiplier", type=int, help="multiplier for boosting viusal",
            default=1)
    parser.add_argument("--codec", help="codec to use", default="mp4v")
    parser.add_argument("--fps", type=int, help="frames per sec for output",
            default=60)

    args = parser.parse_args()

    image_folder = args.image_folder
    video_name = args.video_name
    pattern = args.pattern
    fps = args.fps
    multiplier = args.multiplier
    codec = args.codec

    if pattern: 
        pattern = re.compile(pattern)
        print("using pattern")  

    images = []

    for img in sorted(os.listdir(image_folder), key=natural_keys):
        is_img = img.endswith(".png") or img.endswith(".jpg")
        matches = pattern is None or pattern.match(img) is not None
        if is_img and matches:
            images.append(img)

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name,
                            cv2.VideoWriter_fourcc(*codec), fps,
                            (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)) * multiplier)

    video.release()
    # cv2.destroyAllWindows()
