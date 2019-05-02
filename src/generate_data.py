import os
import sys
import cv2
import argparse
import numpy as np
from utils import draw_bodypose, get_body

# Creates command line parser
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("target", help="path to target video")
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
            h, w, _ = frame.shape
            frame = cv2.resize(frame, (int(w/2), int(h/2)))

            candidate, subset = body_estimation(frame)
            if np.min(subset[:,19]) < 18:
                print('Frame Dropped', frame_counter)
                frame_counter += 1
                continue

            canvas = np.ones((int(h/2), int(w/2), 3), dtype='uint8') * 255
            image = draw_bodypose(canvas, candidate, subset)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
            cv2.imwrite('data/TargetPoses/pose.png' + str(frame_counter), image)
            cv2.imwrite('data/TargetFrames/frame.png' + str(frame_counter))
            frame_counter += 1
            print(frame_counter)
        else:
            break
    video.release()

if __name__ == "__main__":
    main()