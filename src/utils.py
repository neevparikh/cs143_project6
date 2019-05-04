import numpy as np
import math
import cv2
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import logging
import sys
import os
import pickle

sys.path.append(os.getcwd() + '/src/')
sys.path.append(os.getcwd() + '/src/pix2pixHD/')
sys.path.append(os.getcwd() + '/src/pytorch-openpose/')

from python import body

def get_body():
    return body.Body(os.getcwd() + '/src/pytorch-openpose/model/body_pose_model.pth')

class PoseNormalizer:
    ''' Normalizes the pose as described in the Everybody Dance Now paper '''

    def __init__(self, source, target, epsilon=0.7, inclusion_threshold=20):
        """
            source :: dict<ndarray> :: dict of source left ankle array and source right ankle array
            target :: dict<ndarray> :: dict of target left ankle array and target right ankle array
            epsilon :: float [0, 1] :: value for the clustering in calculating the min, paper suggests 0.7  
        """

        self.inclusion_threshold = inclusion_threshold
        self.s_left, self.s_right = self._include_ground_only(source["left"], source["right"]) 
        self.t_left, self.t_right = self._include_ground_only(target["left"], target["right"]) 
        self.epsilon = epsilon
        self.statistics = {}
        self._compute_statistics(np.append(self.s_left, self.s_right), "source")
        self._compute_statistics(np.append(self.t_left, self.t_right), "target")

    
    def _include_ground_only(self, left_ankle_array, right_ankle_array):
        """ remove the frames where the leg is raised """

        num_frames = len(left_ankle_array)
        left_grounded = [] 
        right_grounded = []

        for i in range(num_frames):
            if np.abs(left_ankle_array[i] - right_ankle_array[i]) < self.inclusion_threshold:
                left_grounded.append(left_ankle_array[i])
                right_grounded.append(right_ankle_array[i])
            else:
                pass

        return np.array(left_grounded), np.array(right_grounded)

    def _compute_translation(self, source, target):
        """ b = t_min + (avg_frame_pos_source - s_min) / (s_max - s_min) * (t_max - t_min) - f_source """

        # NOTE: f_source assumed to be avg_target as we don't know what it is yet
        avg_source = (source["left"] + source["right"]) / 2
        avg_target = (target["left"] + target["right"]) / 2
        t_min = self.statistics["target"]["min"]
        t_max = self.statistics["target"]["max"]
        s_min = self.statistics["source"]["min"]
        s_max = self.statistics["source"]["max"]

        return t_min + ((avg_source - s_min) / (s_max - s_min)) * (t_max - t_min) - avg_target # self.statistics["target"]["total_avg"]

    def _compute_scale(self, source):
        """ s = t_far / s_far + (a_source - s_min) / (s_max - s_min) * (t_close / s_close - t_far / s_far) """

        avg_source = (source["left"] + source["right"]) / 2
        t_far = self.statistics["target"]["far"]
        t_close = self.statistics["target"]["close"]
        s_far = self.statistics["source"]["far"]
        s_close = self.statistics["source"]["close"]
        s_min = self.statistics["source"]["min"]
        s_max = self.statistics["source"]["max"]

        return (t_far / s_far) + (avg_source - s_min) / (s_max - s_min) * ((t_close / s_close) - (t_far / s_far))

    def _compute_statistics(self, ankle_array, ankle_name):
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
        self.statistics[ankle_name]["close"], self.statistics[ankle_name]["far"] = self._get_close_far_position(ankle_array, mx, mn)
    
    def _get_median_ankle_position(self, ankle_array):
        return np.median(ankle_array, overwrite_input=False)
    
    def _get_min_ankle_position(self, ankle_array, med, mx):
        cluster = np.array([p for p in ankle_array if (p < med) and (np.abs(np.abs(p - med) - np.abs(mx - med)) < self.epsilon)])
        return np.max(cluster)

    def _get_close_far_position(self, ankle_array, mx, mn):
        cluster_far = np.array([p for p in ankle_array if (np.abs(p - mn) < self.epsilon)])
        cluster_close = np.array([p for p in ankle_array if (np.abs(p - mx) < self.epsilon)])
        return np.max(cluster_close), np.max(cluster_far)

    def _get_max_ankle_position(self, ankle_array):
        return np.max(ankle_array)

    def transform_pose(self, source, target):
        """
            source :: ndarray :: numpy array of all the pose estimates as returned by pose estimation of source video 
            target :: ndarray :: numpy array of all the pose estimates as returned by pose estimation of target video
            
            Returns :: normalized target in the same format 
        """

        source_ankles = {"left": source[13, 1], "right": source[10, 1]}
        target_ankles = {"left": target[13, 1], "right": target[10, 1]}

        b = self._compute_translation(source_ankles, target_ankles)
        s = self._compute_scale(source_ankles)
        source[:, 1] *= s
        source[:, 1] += b
        source[:, 0:2] = source.astype("int")[:, 0:2]
        return source

    def transform_pose_global(self, source_all, target_all):
        """
            source :: list<ndarray> :: numpy array of all the pose estimates for all the frames of the source 
            target :: list<ndarray> :: numpy array of all the pose estimates for all the frames of the target

            Returns :: globally normalized in the same format
        """
        source_ankles = {"left": self.statistics["source"]["total_avg"], "right": self.statistics["source"]["total_avg"]}
        target_ankles = {"left": self.statistics["target"]["total_avg"], "right": self.statistics["target"]["total_avg"]}
        b = self._compute_translation(source_ankles, target_ankles)
        s = self._compute_scale(source_ankles)
        for i in range(len(source_all)):
            p = source_all[i]
            p[:, 1] *= s
            p[:, 1] += b
            p[:, 0:2] = p.astype("int")[:, 0:2]
            source_all[i] = p
        return source_all

def start_run(config_ctr):
    config_obj = config_ctr()
    config_obj.parse()
    config_obj.opt.path = os.path.join("log_dir", "name")
    config = config_obj.opt

    device = torch.device("cuda")

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
        for val, slope, total in zip(self.values[::-1], self.slopes[::-1], self.totals[::-1]):
            if self.iter > total:
                return val + slope * (self.iter - total)
        raise ValueError

def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
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
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    # colors = [[10, 10, 10], [20, 20, 20], [30, 30, 30], [40, 40, 40], [50, 50, 50], [60, 60, 60], [70, 70, 70], \
    #          [80, 80, 80], [80, 80, 80], [90, 90, 90], [100, 100, 100], [110, 110, 110], [120, 120, 120], [130, 130, 130], \
    #          [140, 140, 140], [150, 150, 150], [160, 160, 160], [170, 170, 170]]

    colors = [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7], \
              [8, 8, 8], [8, 8, 8], [9, 9, 9], [10, 10, 10], [11, 11, 11], [12, 12, 12], [13, 13, 13], \
              [14, 14, 14], [15, 15, 15], [16, 16, 16], [17, 17, 17]]

    for i in range(17):
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
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cur_canvas
    
    return canvas

def get_pose_estimate(video_location, regen=True, rotate=True):
    """
    video_location :: location of video
    regen :: to regenerate or use pickles
    Returns python list of poses for the video
    """

    if os.path.isfile(video_location): 
        video = cv2.VideoCapture(video_location)
    else: 
        raise FileNotFoundError 

    body_estimation = body.Body('pytorch-openpose/model/body_pose_model.pth')

    # Initialize arrays to store pose information
    poses = []
    subsets = []

    # Frame counter
    frame_counter = 0

    if regen:
        while(True): 
            # reading from frame 
            ret, frame = video.read()

            if ret:
                if rotate:
                    frame = np.rot90(np.rot90(np.rot90(frame)))
                h, w, _ = frame.shape
                frame = cv2.resize(frame, (int(w/2), int(h/2)))

                # Grab pose estimations for both video frames
                candidate, subset = body_estimation(frame)

                if np.min(subset[:,19]) < 18:
                    print('Frame Dropped', frame_counter)
                    frame_counter += 1
                    continue

                # Put pose estimations into memory
                poses.append(candidate)
                subsets.append(subset)

                frame_counter += 1
                print("Frame: ", frame_counter)
            else:
                np.save("poses.npy", np.array(poses))
                np.save("subsets.npy", np.array(subsets))
                break

        video.release()
    else:
        poses = np.load("poses.npy", allow_pickle=True)
        subsets = np.load("subsets.npy", allow_pickle=True)

    return poses, subsets

def get_pose_normed_estimate(source, target, regen=True, rotate=True, height=512, width=256):
    """
    source :: location of source video
    target :: location of target video
    regen :: to regenerate or use pickles
    Returns numpy array of normalized poses for the source
    """
    if os.path.isfile(source) and os.path.isfile(target):
        source_video = cv2.VideoCapture(source)
        target_video = cv2.VideoCapture(target)
    else: 
        raise FileNotFoundError 

    body_estimation = body.Body('pytorch-openpose/model/body_pose_model.pth')

    # Initialize arrays to store pose information
    source_poses = []
    target_poses = []
    source_subsets = []
    target_subsets = []

    # Initialize arrays for PoseNormalizer
    source_left = []
    source_right = []
    target_left = []
    target_right = []
    source_frames = []
    target_frames = []

    # Frame counter
    frame_counter = 0

    if regen:
        while(True): 
            # reading from frame 
            ret_source, source_frame = source_video.read() 
            ret_target, target_frame = target_video.read()

            if ret_source and ret_target:
                if rotate:
                    source_frame = np.rot90(np.rot90(np.rot90(source_frame)))
                    target_frame = np.rot90(np.rot90(np.rot90(target_frame)))

                source_frame = cv2.resize(source_frame, (int(width), int(height)))
                target_frame = cv2.resize(target_frame, (int(width), int(height)))
                
                source_frames.append(source_frame)
                target_frames.append(target_frame)

                # Grab pose estimations for both video frames
                source_candidate, source_subset = body_estimation(source_frame)
                target_candidate, target_subset = body_estimation(target_frame)

                if np.min(source_subset[:,19]) < 18 or np.min(target_subset[:,19]) < 18:
                    print('Frame Dropped', frame_counter)
                    frame_counter += 1
                    continue

                # Put pose estimations into memory
                source_poses.append(source_candidate)
                target_poses.append(target_candidate)

                source_subsets.append(source_subset)
                target_subsets.append(target_subset)

                # Grab ankles
                source_left.append(source_candidate[13, 1])
                source_right.append(source_candidate[10, 1])
                target_left.append(target_candidate[13, 1])
                target_right.append(target_candidate[10, 1])

                frame_counter += 1
                print("Frame: ", frame_counter)
            else:
                pickle.dump(source_poses, open("source_poses.pkl", "wb"))
                pickle.dump(target_poses, open("target_poses.pkl", "wb"))
                pickle.dump(source_subsets, open("source_subsets.pkl", "wb"))
                pickle.dump(target_subsets, open("target_subsets.pkl", "wb"))
                break

        source_video.release()
        target_video.release()


        source_dict = {
        "left": np.array(source_left),
        "right": np.array(source_right)
        }

        target_dict = {
        "left": np.array(target_left),
        "right": np.array(target_right)
        }

        pose_normalizer = PoseNormalizer(source_dict, target_dict, epsilon=3)
        transformed_all = pose_normalizer.transform_pose_global(source_poses, target_poses)
        np.save("normed_source_pose.npy", np.array(transformed_all))

    else:
        source_poses = pickle.load(open("source_poses.pkl", "rb"))
        target_poses = pickle.load(open("target_poses.pkl", "rb"))
        source_subsets = pickle.load(open("source_subsets.pkl", "rb"))
        target_subsets = pickle.load(open("target_subsets.pkl", "rb"))
        transformed_all = np.load("normed_source_pose.npy")

    data = {
        "normed_source": transformed_all, 
        "source_pose": source_poses, 
        "source_sub": source_subsets, 
        "target_pose": target_poses, 
        "target_sub": target_subsets, 
        "source_frame": source_frames, 
        "target_frame": target_frames
    }
   
    return data
