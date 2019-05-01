import cv2 
import numpy as np 
import sys
sys.path.append("/home/neev/repos/cs143_project6/src/pytorch-openpose/python/")
from util import draw_bodypose  

class GANWrapper:
    ''' GAN wrapper to interface with pipeline. '''

    def __init__(self, source, target, mode="transfer"):
        self.mode = mode
        self.source = source 
        self.target = target 

    def create_image_from_pose(self, poses, subsets):
        """ """	
        images = []
        for pose,subset in zip(poses, subsets):
            canvas = np.ones((480, 720, 3), dtype='uint8')
            image = draw_bodypose(canvas, pose, subset)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, image = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)
            images.append(image)
        return images


    def apply_mapping(self, norm_source_poses, source_subsets):
        return norm_poses

    def train(self):
        pass
