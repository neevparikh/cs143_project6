import numpy as np
import os
import cv2
from utils import transform_frame, save_dir
from generate_data import add_base_args
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="path to target video", default=None)

    add_base_args(parser)

    args = parser.parse_args()

    source_indexes = np.load(
        '{}/source_indexes.npy'.format(save_dir), allow_pickle=True)

    parser.add_argument("source", help="path to target video", default=None)

    video = cv2.VideoCapture(args.source)

    counter = 0

    frame = None

    def get_frame():
        nonlocal frame
        ret, frame = video.read()
        return ret

    while (get_frame()):
        if counter in source_indexes:
            cv2.imwrite(os.path.join('data/test_img',
                                     'img_{}.png'.format(counter)),
                        transform_frame(frame, args.rotated, args.width,
                                        args.height))
            print('source frame {} written'.format(counter))
        else:
            print('source frame {} dropped'.format(counter))
        counter += 1

    video.release()


if __name__ == '__main__':
    main()
