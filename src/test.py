import utils
from data.data_loader import CreateDataLoader
from options.test_options import TestOptions
from models.models import create_model
import os
from cv2 import imwrite

def generate(config, writer, logger):
    config = config.opt
    data_set = CreateDataLoader(config).load_data()
    model = create_model(config)

    for i, data in enumerate(data_set):
        minibatch = 1
        generated = model.inference(data['label'], data['inst'])
        imwrite(os.path.join("outputs", "output_{}.png".format(i)), generated)

if __name__ == '__main__':
    generate(*utils.start_run(TestOptions))
