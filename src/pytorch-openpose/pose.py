import sys
import os
sys.path.insert(0, 'pytorch-openpose/python')
sys.path.insert(0, 'pytorch-openpose')
import cv2
import model
import util
from hand import Hand
from body import Body
from pose_norm import PoseNormalizer
import matplotlib.pyplot as plt
import copy
import pickle
import numpy as np

# TODO:

def get_pose_estimate(video_location, regen=True):
    """
    video_location :: location of video
    regen :: to regenerate or use pickles
    Returns python list of poses for the video
    """

    if os.path.isfile(video_location): 
        video = cv2.VideoCapture(video_location)
    else: 
        raise FileNotFoundError 

    body_estimation = Body('pytorch-openpose/model/body_pose_model.pth')

    # Initialize arrays to store pose information
    poses = []

    # Frame counter
    frame_counter = 0

    if regen:
        while(True): 
            # reading from frame 
            ret, frame = video.read()

            if ret:
                frame = np.rot90(np.rot90(np.rot90(frame)))
                h, w, _ = frame.shape
                frame = cv2.resize(frame, (int(w/2), int(h/2)))

                # Grab pose estimations for both video frames
                candidate, _ = body_estimation(frame)

                # Put pose estimations into memory
                poses.append(candidate)

                frame_counter += 1
                print("Frame: ", frame_counter)
            else:
                pickle.dump(poses, open("poses.pkl", "wb"))
                break

        video.release()
    else:
        poses = pickle.load(open("poses.pkl", "rb"))

    return poses

def get_pose_normed_estimate(source, target, regen=True):
    """
    source :: location of source video
    target :: location of target video
    regen :: to regenerate or use pickles
    Returns python list of normalized poses for the source
    """
    if os.path.isfile(source) and os.path.isfile(target):
        source_video = cv2.VideoCapture(source)
        target_video = cv2.VideoCapture(target)
    else: 
        raise FileNotFoundError 

    body_estimation = Body('pytorch-openpose/model/body_pose_model.pth')

    # Initialize arrays to store pose information
    source_poses = []
    target_poses = []

    # Initialize arrays for PoseNormalizer
    source_left = []
    source_right = []
    target_left = []
    target_right = []

    # Frame counter
    frame_counter = 0

    if regen:
        while(True): 
            # reading from frame 
            ret_source, source_frame = source_video.read() 
            ret_target, target_frame = target_video.read()

            if ret_source and ret_target:
                source_frame = np.rot90(np.rot90(np.rot90(source_frame)))
                h, w, _ = source_frame.shape
                source_frame = cv2.resize(source_frame, (int(w/2), int(h/2)))

                target_frame = np.rot90(np.rot90(np.rot90(target_frame)))
                target_frame = cv2.resize(target_frame, (int(w/2), int(h/2)))

                # Grab pose estimations for both video frames
                source_candidate, _ = body_estimation(source_frame)
                target_candidate, _ = body_estimation(target_frame)

                # Put pose estimations into memory
                source_poses.append(source_candidate)
                target_poses.append(target_candidate)

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
                pickle.dump(source_left, open("source_left.pkl", "wb"))
                pickle.dump(source_right, open("source_right.pkl", "wb"))
                pickle.dump(target_left, open("target_left.pkl", "wb"))
                pickle.dump(target_right, open("target_right.pkl", "wb"))
                break
    else:
        source_poses = pickle.load(open("source_poses.pkl", "rb"))
        target_poses = pickle.load(open("target_poses.pkl", "rb"))
        source_left = pickle.load(open("source_left.pkl", "rb"))
        source_right = pickle.load(open("source_right.pkl", "rb"))
        target_left = pickle.load(open("target_left.pkl", "rb"))
        target_right = pickle.load(open("target_left.pkl", "rb"))


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
    
    return transformed_all, source_poses, target_poses
