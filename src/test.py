import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict
import cv2
import utils
from torch.autograd import Variable 
import util.util as util
from data.data_loader import CreateDataLoader
from util.visualizer import Visualizer
from models.models import create_model
from util import html


#TODO
# Load the model 
# Evaluate the model again
# Write the files or something ?

def generate(writer, config, logger):
    config = config.opt
    data_set = CreateDataLoader(config).load_data()
    model = create_model(config)

    for i, data in enumerate(data_set):
        minibatch = 1 
        generated = model.inference(data['label'], data['inst'])
        imwrite( "../outputs/output_" + str(i) + ".png", generated)      
    torch.cuda.empty_cache()
