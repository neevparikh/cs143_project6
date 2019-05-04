import utils
import numpy as np
from data.data_loader import CreateDataLoader
from options.test_options import TestOptions
from models.models import create_model
import os
from torchvision.transforms import ToPILImage
from cv2 import imwrite

def generate(config, writer, logger):
    config = config.opt
    data_set = CreateDataLoader(config).load_data()
    model = create_model(config)

    for i, data in enumerate(data_set):
        minibatch = 1
        generated = model.inference(data['label'], data['inst'])
        generated_img = generated.detach().cpu().numpy()[0]
        generated_img = np.moveaxis(generated_img, [0, 1, 2], [2, 1, 0])
        for i == 0:
            print(np.max(generated_img))
        imwrite(os.path.join("outputs", "output_{}.png".format(i)), generated_img)

if __name__ == '__main__':
    generate(*utils.start_run(TestOptions))
