import logging
import math
import os
import pickle
import re
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tensorboardX import SummaryWriter


sys.path.append(os.getcwd() + '/src/')
sys.path.append(os.getcwd() + '/src/pix2pixHD/')
sys.path.append(os.getcwd() + '/src/pytorch-openpose/')

from data.base_dataset import get_params, get_transform
from python import body

save_dir = "pose_data/"

os.makedirs(save_dir, exist_ok=True)
os.makedirs('data/train_label', exist_ok=True)
os.makedirs('data/train_img', exist_ok=True)

os.makedirs('data/test_label', exist_ok=True)
os.makedirs('data/test_img', exist_ok=True)
os.makedirs('data/test_inst', exist_ok=True)


def load_average_img(config):
    average_img = Image.open("data/average.png").convert('RGB')
    params = get_params(config, average_img.size)
    transform = get_transform(config, params)
    average_tensor = transform(average_img)

    return average_tensor


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def get_ordered_files(path):
    return sorted(os.listdir(path), key=natural_keys)


def get_body():
    return body.Body(os.getcwd() +
                     '/src/pytorch-openpose/model/body_pose_model.pth')


def chunk(xs, n):
    '''Split the list, xs, into n chunks'''
    L = len(xs)
    assert 0 < n <= L
    s, r = divmod(L, n)
    chunks = [xs[p:p+s] for p in range(0, L, s)]
    chunks[n-1:] = [xs[-r-s:]]
    return chunks


class PoseNormalizer:
    ''' Normalizes the pose as described in the Everybody Dance Now paper '''

    def __init__(self, source, target, epsilon=0.7, inclusion_threshold=20,
                 alpha=1, manual_scale=None, manual_translate=None):
        """
        source :: dict<ndarray> :: dict of source left ankle array and
                                   source right ankle array
        target :: dict<ndarray> :: dict of target left ankle array and
                                   target right ankle array
        epsilon :: float [0, 1] :: value for the clustering in calculating
                                   the min, paper suggests 0.7
        """

        self.inclusion_threshold = inclusion_threshold
        self.s_left, self.s_right = self._include_ground_only(
            source["left"], source["right"])
        self.t_left, self.t_right = self._include_ground_only(
            target["left"], target["right"])
        self.epsilon = epsilon
        self.alpha = alpha
        self.manual_scale = manual_scale
        self.manual_translate = manual_translate
        self.statistics = {}
        self._compute_statistics(
            np.append(self.s_left, self.s_right), "source")
        self._compute_statistics(
            np.append(self.t_left, self.t_right), "target")

    def _include_ground_only(self, left_ankle_array, right_ankle_array):
        """ remove the frames where the leg is raised """

        num_frames = len(left_ankle_array)
        ground_lvl = (np.mean(left_ankle_array[0:5]) +
                      np.mean(right_ankle_array[0:5])) // 2
        left_grounded = []
        right_grounded = []

        for i in range(num_frames):
            if np.abs(left_ankle_array[i]
                      - right_ankle_array[i]) < self.inclusion_threshold:
                left_grounded.append(left_ankle_array[i])
                right_grounded.append(right_ankle_array[i])
            else:
                diff = np.abs(left_ankle_array[i]
                      - right_ankle_array[i])
                print('leg raised above ground, not including. diff:',
                      diff, "inclusion_threshold:", self.inclusion_threshold)

        return np.array(left_grounded), np.array(right_grounded)

    def _compute_translation(self, source, target):
        """ b = t_min + (avg_frame_pos_source - s_min) / (s_max - s_min) *
                (t_max - t_min) - f_source """

        if self.manual_translate is not None:
            return self.manual_translate
        else:
            # NOTE: f_source assumed to be avg_target as we don't know what it is
            # yet

            avg_source = (source["left"] + source["right"]) / 2
            avg_target = (target["left"] + target["right"]) / 2
            t_min = self.statistics["target"]["min"]
            t_max = self.statistics["target"]["max"]
            s_min = self.statistics["source"]["min"]
            s_max = self.statistics["source"]["max"]

            # self.statistics["target"]["total_avg"]
            return t_min + ((avg_source - s_min) / (s_max - s_min)) * \
                (t_max - t_min) - (self.alpha * avg_target)

    def _compute_scale(self, source):
        """ s = t_far / s_far + (a_source - s_min) / (s_max - s_min) *
        (t_close / s_close - t_far / s_far) """
        if self.manual_scale is not None:
            return self.manual_scale
        else:
            avg_source = (source["left"] + source["right"]) / 2
            t_far = self.statistics["target"]["far"]
            t_close = self.statistics["target"]["close"]
            s_far = self.statistics["source"]["far"]
            s_close = self.statistics["source"]["close"]
            s_min = self.statistics["source"]["min"]
            s_max = self.statistics["source"]["max"]

            return (t_far / s_far) + ((avg_source - s_min) /
                   (s_max - s_min)) * ((t_close / s_close) - (t_far / s_far))

    def _compute_statistics(self, ankle_array, ankle_name):
        print('estimating statistics for:', ankle_name)
        med = self._get_median_ankle_position(ankle_array)
        mx = self._get_max_ankle_position(ankle_array)
        avg = np.average(ankle_array)
        self.statistics[ankle_name] = {
            "med": med,
            "max": mx,
            "total_avg": avg
        }
        mn = self._get_min_ankle_position(ankle_array, med, mx)
        self.statistics[ankle_name]["min"] = mn

        close_far = self._get_close_far_position(ankle_array, mx, mn)
        self.statistics[ankle_name]["close"] = close_far[0]
        self.statistics[ankle_name]["far"] = close_far[1]

        print('statistics estimated:', self.statistics)

    def _get_median_ankle_position(self, ankle_array):
        return np.median(ankle_array, overwrite_input=False)

    def _get_min_ankle_position(self, ankle_array, med, mx):
        try:
            cluster = np.array([p for p in ankle_array if (p < med) and (
                np.abs(np.abs(p - med) - np.abs(mx - med)) < self.epsilon)])
            print('size of cluster:', cluster.shape[0])
            mn = np.max(cluster)
        except Exception as e:
            print("Warning: Minimum as defined failed, reverting to stable alg")
            mn_est = np.abs(med - np.abs(mx - med))
            print("estimated min:", mn_est, " finding closest")
            mn = ankle_array[(np.abs(ankle_array - mn_est)).argmin()]
        return mn

    def _get_close_far_position(self, ankle_array, mx, mn):
        cluster_far = np.array(
            [p for p in ankle_array if np.abs(p - mn) < self.epsilon])
        cluster_close = np.array(
            [p for p in ankle_array if np.abs(p - mx) < self.epsilon])
        return np.max(cluster_close), np.max(cluster_far)

    def _get_max_ankle_position(self, ankle_array):
        return np.max(ankle_array)

    def transform_pose_global(self, source_all):
        """
        source :: list<ndarray> :: numpy array of all the pose estimates
                                   for all the frames of the source
        target :: list<ndarray> :: numpy array of all the pose estimates
                                   for all the frames of the target

        Returns :: globally normalized in the same format
        """

        source_ankles = {
            "left": self.statistics["source"]["total_avg"],
            "right": self.statistics["source"]["total_avg"]
        }
        target_ankles = {
            "left": self.statistics["target"]["total_avg"],
            "right": self.statistics["target"]["total_avg"]
        }
        b = self._compute_translation(source_ankles, target_ankles)
        s = self._compute_scale(source_ankles)
        print('translation and scale:', b, s)

        for i in range(len(source_all)):
            p = source_all[i]
            max_v = p[:, 1].max()
            p[:, 1] -= max_v
            p[:, 1] *= s
            p[:, 1] += max_v
            p[:, 1] += b
            p[:, 0:2] = p.astype("int")[:, 0:2]
            source_all[i] = p
        return source_all


def start_run(config_ctr):
    config_obj = config_ctr()
    config_obj.parse()
    config_obj.opt.path = os.path.join("log_dir", "name")
    config = config_obj.opt

    # tensorboard (TODO)
    # writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
    # writer.add_text('config', config.as_markdown(), 0)

    logger = get_logger(os.path.join(
        config.path, "{}.log".format(config.name)))
    # config.print_params(logger.info)

    logger.info("Logger is set - training start")

    # set seed (TODO)
    # np.random.seed(config.seed)
    # torch.manual_seed(config.seed)
    # torch.cuda.manual_seed_all(config.seed)

    torch.backends.cudnn.benchmark = True

    # return config_obj, writer, logger
    return config_obj, None, logger


class CustomSchedule:
    def __init__(self, num_iters, param_values, intervals):
        assert len(param_values) == len(intervals) + 1

        sum_v = 0.
        self.totals = []
        self.slopes = []
        prev_val = param_values[0]
        for val, interv in zip(param_values[1:], intervals):
            self.totals.append(sum_v)
            interv *= num_iters
            sum_v += interv
            slope = (val - prev_val) / interv
            prev_val = val
            self.slopes.append(slope)

        assert abs(sum_v - num_iters) < 1e-6

        self.iter = 0
        self.values = param_values[:-1]
        self.num_iters = num_iters

    def calc(self):
        self.iter += 1
        if self.iter > self.num_iters:
            self.iter = self.num_iters
        for val, slope, total in zip(self.values[::-1], self.slopes[::-1],
                                     self.totals[::-1]):
            if self.iter > total:
                return val + slope * (self.iter - total)
        raise ValueError


def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()),
    # we should use custom logger
    logger = logging.getLogger('darts')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


class AverageMeter():
    """ Computes and stores the average and current value """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def draw_bodypose(canvas, pose, subset):
    """
    canvas :: canvas to be drawn on
    pose :: pose of person
    subset :: subset data of the person
    """
    stickwidth = 4
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6],
              [7, 7, 7], [8, 8, 8], [8, 8, 8], [9, 9, 9], [10, 10, 10],
              [11, 11, 11], [12, 12, 12], [13, 13, 13], [14, 14, 14],
              [15, 15, 15], [16, 16, 16], [17, 17, 17]]

    num_pose_points = 17

    for i in range(num_pose_points):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = pose[index.astype(int), 0]
            X = pose[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)),
                                       (int(length / 2), stickwidth),
                                       int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cur_canvas

    return canvas
