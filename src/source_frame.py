import numpy as np 
import os 
import cv2
from utils import transform_frame

rotated = True
width = 512
height = 1024
vid_path = os.path.join('data/videos/neev.mp4')
path = 'data/test_img/'
source_indexes = np.load('pose_data/source_indexes.npy', allow_pickle=True)
vid = cv2.VideoCapture(vid_path)
counter = 0 

while(True):
    ret, frame = vid.read()
    if ret:
        if counter in source_indexes:
            cv2.imwrite(os.path.join(path, 'img_{}.png'.format(counter)),
                    transform_frame(frame, rotated, width, height))
            print('source frame {} written'.format(counter))
        else:
            print('source frame {} dropped'.format(counter))
        counter += 1 
    else:
        break

vid.release()
        
    

