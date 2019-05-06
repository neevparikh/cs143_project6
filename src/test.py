import utils
import numpy as np
from data.data_loader import CreateDataLoader
from options.test_options import TestOptions
from models.models import create_model
import os
from torchvision.transforms import ToPILImage
from cv2 import imwrite
from tqdm import tqdm
import util.util as util
from collections import OrderedDict
from util.visualizer import Visualizer
from util import html

def generate(config, writer, logger):
    config = config.opt
    data_set = CreateDataLoader(config).load_data()
    model = create_model(config)
    visualizer = Visualizer(config)

    web_dir = os.path.join(config.results_dir, config.name, '%s_%s' %
                           (config.phase, config.which_epoch))

    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' %
                        (config.name, config.phase, config.which_epoch))

    for data in tqdm(data_set):
        data['label'] = data['label'][:,:1]
        assert data['label'].shape[1] == 1

        generated = model.inference(data['label'], data['inst'])
        visuals = OrderedDict([('input_label',
                                util.tensor2label(data['label'][0],
                                                   config.label_nc)),
                               ('synthesized_image',
                                util.tensor2im(generated.data[0]))])

        img_path = data['path']
        visualizer.save_images(webpage, visuals, img_path)

if __name__ == '__main__':
    generate(*utils.start_run(TestOptions))
