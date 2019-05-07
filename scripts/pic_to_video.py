#!/usr/bin/env python3

import cv2
import os
import sys
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
    image_folder = sys.argv[1]
    video_name = sys.argv[2]
    pattern = None

    if len(sys.argv) > 3:
        pattern = re.compile(sys.argv[3])
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
                            cv2.VideoWriter_fourcc('M', 'P', '4', '2'), 60,
                            (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()
