import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict

import utils
from options.train_options import TrainOptions
import util.util as util
from data.data_loader import CreateDataLoader
from util.visualizer import Visualizer
from models.models import create_model

#TODO:
# - Logging?
# - TensorBoard?


def train(config, writer, logger):
    config = config.opt
    data_set = CreateDataLoader(config).load_data()
    total_steps = config.epochs * len(data_set)
    print(len(data_set), "# of Training Images")

    model = create_model(config)
    visualizer = Visualizer(config)

    step = 0

    for epoch in range(config.epochs):
        print("epoch: ", epoch)
        for data in data_set:
            save_gen = (step + 1) % config.display_freq == 0
            # save_gen = True

            losses, generated = model(Variable(data['label']),
                                      Variable(data['inst']),
                                      Variable(data['image']),
                                      Variable(data['feat']),
                                      infer=save_gen)

            # sum per device losses
            losses = [torch.mean(x) if not isinstance(
                x, int) else x for x in losses]
            loss_dict = dict(zip(model.module.loss_names, losses))

            # calculate final loss scalar
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
            loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat', 0) + \
                loss_dict.get('G_VGG', 0)

            # update generator weights\n",
            model.module.optimizer_G.zero_grad()
            loss_G.backward()
            model.module.optimizer_G.step()

            # update discriminator weights\n",
            model.module.optimizer_D.zero_grad()
            loss_D.backward()
            model.module.optimizer_D.step()

            if (step + 1) % config.print_freq == 0 or step == total_steps - 1:
                logger.info("Train: [{:2d}/{}] Step {:03d}/{:03d}".format(
                    epoch + 1, config.epochs, step, len(data_set) - 1))
                logger.info("Loss D: {},  Loss G {}, Loss VGG {}".format(
                    loss_D.item(), loss_dict['G_GAN'].item(), loss_dict['G_VGG'].item()))

            if save_gen:
                visuals = OrderedDict([('input_label',
                                        util.tensor2label(data['label'][0],
                                                          config.label_nc)),
                                       ('synthesized_image', util.tensor2im(
                                           generated.data[0])),
                                       ('real_image',
                                        util.tensor2im(data['image'][0]))])

                visualizer.display_current_results(visuals, epoch, total_steps)


            if (step + 1) % config.save_latest_freq == 0 or step == total_steps - 1:
                model.module.save('latest')
                model.module.save(epoch)

            step += 1

        ### train the entire network after certain iterations
        if (config.niter_fix_global != 0) and (epoch == config.niter_fix_global):
            model.module.update_fixed_params()

        ### linearly decay learning rate after certain iterations
        if epoch > config.niter:
            model.module.update_learning_rate()

if __name__ == '__main__':
    train(*utils.start_run(TrainOptions))
