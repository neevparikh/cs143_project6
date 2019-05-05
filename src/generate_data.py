import cv2
import numpy as np

from utils import draw_bodypose


def add_base_args(parser):
    parser.add_argument("--height", type=int, default=1024,
                        help="height for saved video")
    parser.add_argument("--width", type=int, default=512,
                        help="width for saved video")
    parser.add_argument("--rotated", choices=["true", "false"], required=True)
    parser.add_argument("--max_frames", type=float, default=float('inf'))


def make_get_path(is_train, is_label):
    type_name = 'label' if is_label else 'img'
    path_b = "data/{}_{}/{}_".format(('train' if is_train else 'test'),
                                     type_name, type_name)

    def get_path(counter, path_b=path_b):
        return "{}{}.png".format(path_b, counter)

    return get_path


def save_pose(pose, subset, path, height, width):
    canvas = np.ones((height, width, 3), dtype='uint8')
    target_image = draw_bodypose(canvas, pose, subset)
    cv2.imwrite(path, target_image)
