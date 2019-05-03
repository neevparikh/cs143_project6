import os
import sys
import cv2
import argparse
import numpy as np
from utils import draw_bodypose, get_body

def main():
    # Creates command line parser
    parser = argparse.ArgumentParser()

    parser.add_argument("target", help="path to target video")
    parser.add_argument("--height", type=int, default=1024,
                        help="height for saved video")
    parser.add_argument("--width", type=int, default=512,
                        help="width for saved video")
    parser.add_argument("--rotated", choices=["true", "false"], required=True)

    args = parser.parse_args()
    target = args.target
    rotated = args.rotated

    if os.path.isfile(target):
        video = cv2.VideoCapture(target)
    else:
        raise FileNotFoundError

    frame_counter = 0

    body_estimation = get_body()

    while(True):
        ret, frame = video.read()
        if ret:
            if rotated == "true":
                frame = np.rot90(np.rot90(np.rot90(frame)))
            frame = cv2.resize(frame, (args.width, args.height))

            candidate, subset = body_estimation(frame)
            try:
                if np.min(subset[:,19]) < 17:
                    print('Frame Dropped', frame_counter, np.min(subset[:,19]))
                    frame_counter += 1
                    continue
            except Exception as e:
                print(e)
                frame_counter += 1
                continue

            canvas = np.ones((args.height, args.width, 3), dtype='uint8')
            image = draw_bodypose(canvas, candidate, subset)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imwrite('data/train_label/pose_' + str(frame_counter) + '.png', image)
            cv2.imwrite('data/train_img/frame_' + str(frame_counter) + '.png', frame)
            frame_counter += 1
            print(frame_counter, ' Written')
        else:
            break
    video.release()

if __name__ == "__main__":
    main()
