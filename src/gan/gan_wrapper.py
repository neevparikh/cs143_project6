import cv2 
import numpy as np 


class GANWrapper:
    ''' GAN wrapper to interface with pipeline. '''

    def __init__(self, source, target, mode="transfer"):
        self.mode = mode
        self.source = source 
        self.target = target 

    def apply_mapping(self, norm_poses):
        return norm_poses

    def train(self):
        pass
