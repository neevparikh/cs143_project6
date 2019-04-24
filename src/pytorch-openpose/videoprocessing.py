import sys
sys.path.insert(0, 'python')
import cv2
import model
import util
from hand import Hand
from body import Body
from pose_norm import PoseNormalizer
import matplotlib.pyplot as plt
import copy
import numpy as np

body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')

source = 'video/SampleDance.mp4'
source_video = cv2.VideoCapture(source)

# TODO Add target video later
target = 'video/SampleDance.mp4'
target_video = cv2.VideoCapture(target)

# Initialize arrays to store pose information
source_poses = []
target_poses = []

# Initialize array to store subset information
target_subsets = []

# Initialize arrays for PoseNormalizer
source_left = []
source_right = []
target_left = []
target_right = []

# Debugging purposes
frame = 0

while(True): 
    # reading from frame 
    ret_source, source_frame = source_video.read() 
    ret_target, target_frame = target_video.read()
    
    if ret_source and ret_target and frame < 60:
      # Resize frames to make computation faster
      source_frame = cv2.resize(source_frame, (720, 480))
      target_frame = cv2.resize(target_frame, (720, 480))

      # Grab pose estimations for both video frames
      source_candidate, source_subset = body_estimation(source_frame)
      target_candidate, target_subset = body_estimation(target_frame)

      # Put target subset into memory
      target_subsets.append(target_subset)

      # Put pose estimations into memory
      source_poses.append(source_candidate)
      target_poses.append(target_candidate)

      # Grab ankles
      source_left.append(source_candidate[13, 1])
      source_right.append(source_candidate[10, 1])
      target_left.append(target_candidate[13, 1])
      target_right.append(target_candidate[10, 1])

      frame += 1
      print(frame)
    else: 
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

pose_normalizer = PoseNormalizer(source_dict, target_dict, epsilon=0.7)

norm_target_poses = []

# Testing video output
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
video=cv2.VideoWriter('testingvideo.mp4', fourcc, 30,(720, 480))

for i in range(len(source_poses)):
  pose = pose_normalizer.transform_pose(source_poses[i], target_poses[i])
  norm_target_poses.append(pose)
  
  # Visually testing if normalizing works 
  canvas = np.ones((480, 720, 3), dtype='uint8') * 255
  canvas = util.draw_bodypose(canvas, pose, target_subsets[i])

video.write(canvas)

# success,image = source_video.read()
# currentFrame = 0
# fourcc = cv2.VideoWriter_fourcc(*'DIVX')
# video=cv2.VideoWriter('video.mp4', fourcc, 30,(720, 480))
