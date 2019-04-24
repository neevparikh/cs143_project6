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

video = 'SampleDance.mp4'
cam = cv2.VideoCapture(video)

success,image = cam.read()
currentFrame = 0
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
video=cv2.VideoWriter('video.mp4', fourcc, 30,(720, 480))

while(True): 
    # reading from frame 
    ret,frame = cam.read() 
  
    if ret:
      frame = cv2.resize(frame, (720, 480))
      candidate, subset = body_estimation(frame)
      
      canvas = copy.deepcopy(frame)
      canvas = np.ones(canvas.shape, dtype='uint8') * 255
      canvas = util.draw_bodypose(canvas, candidate, subset)

      video.write(canvas)
      currentFrame += 1
      print(currentFrame)
    else: 
        break

video.release()
