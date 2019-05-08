import numpy as np
import cv2
from utils import draw_bodypose


src_p = 'pose_data/source_poses.npy'
src_s = 'pose_data/source_subsets.npy'

source_poses = np.load(src_p)
source_subsets = np.load(src_s)

print(source_poses)


# Testing video output
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
video=cv2.VideoWriter('sources.mp4', fourcc, 30,(512, 1024))
frames = source_poses.shape[0]
for i in range(frames):
    # Visually testing if normalizing works 
    canvas = np.ones((1024, 512, 3), dtype='uint8') * 255
    canvas = draw_bodypose(canvas, source_poses[i], source_subsets[i])
    video.write(canvas)
video.release()
