import os
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

    def create_image_from_pose(self, poses, subsets, size=(480, 720, 3)):
        """ """	
        images = []
        for pose,subset in zip(poses, subsets):
            canvas = np.ones(size, dtype='uint8')
            image = draw_bodypose(canvas, pose, subset)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, image = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)
            images.append(image)
        return images

    def create_training_data(self, source_video, source_poses, source_subsets, save_dir):
        """ """ 
        source_images = self.get_raw_images(source_video, rotate=True)
        pose_images = self.create_image_from_pose(source_poses, source_subsets) 

        for i, s_img, p_img in enumerate(zip(source_images, pose_images)):
            concat_img = np.concatenate((s_img, p_img)) 
            print("Saving image ", i)
            cv2.imwrite(save_dir + "image " + str(i), concat_img)



    def get_raw_images(self, source_video, rotate=True, size=(480, 720, 3):
        """ """
        if os.path.isfile(source_video): 
            video = cv2.VideoCapture(source_video)
        else: 
            raise FileNotFoundError 

        images = []
        while(True): 
            # reading from frame 
            ret, frame = video.read()
            if ret:
                if rotate:
                    frame = np.rot90(np.rot90(np.rot90(frame)))
                frame = cv2.resize(frame, size)
                images.append(frame)
            else:
                break
        video.release()
        return images
        


    def apply_mapping(self, norm_source_poses, source_subsets):
        return norm_poses

    def train(self):
        pass
