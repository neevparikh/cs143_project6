from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import utils
import util.util as util
from data.data_loader import CreateDataLoader
from models.models import create_model
from options.train_options import TrainOptions
from util.visualizer import Visualizer
import os

#TODO:
# - Logging?
# - TensorBoard?


def train(config, writer, logger):
    config = config.opt

    if 'WORLD_SIZE' in os.environ:
        config.distributed = int(os.environ['WORLD_SIZE']) > 1
    else:
        config.distributed = 0

    if config.distributed:
        config.gpu = config.local_rank
        torch.cuda.set_device(config.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        config.world_size = torch.distributed.get_world_size()

    data_set = CreateDataLoader(config).load_data()
    total_steps = config.epochs * len(data_set)
    print(len(data_set), "# of Training Images")

    model = create_model(config)
    visualizer = Visualizer(config)

    model.named_buffers = lambda: []

    config.distributed = False

    config.gpu = 0
    config.world_size = 1

    if config.fp16:
        from apex import amp
        from apex.parallel import DistributedDataParallel as DDP
        model = model.cuda()
        model, [optimizer_G, optimizer_D] = \
            amp.initialize(model, [model.optimizer_G, model.optimizer_D],
                           opt_level='O1')

        if config.distributed:
            model = DDP(model)
        else:
            model = torch.nn.DataParallel(model)

    step = 0

    for epoch in range(config.epochs):
        print("epoch: ", epoch)
        for i, data in enumerate(data_set):
            save_gen = (step + 1) % config.display_freq == 0

            if i == 0:
                if config.no_temporal_smoothing:
                    prev_generated = prev_label = prev_real = None
                else:
                    prev_generated = torch.zeros_like(data['image'])
                    prev_label = torch.zeros_like(data['label'])
                    prev_real = torch.zeros_like(data['image'])

            data['label'] = data['label'][:, :1]
            assert data['label'].shape[1] == 1

            losses, generated = model(Variable(data['label']),
                                      Variable(data['inst']),
                                      Variable(data['image']),
                                      Variable(data['feat']),
                                      prev_label,
                                      prev_generated,
                                      prev_real,
                                      infer=save_gen)

            prev_real = data['image'].detach()
            prev_label = data['label'].detach()

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

            if config.fp16:
                with amp.scale_loss(loss_G, optimizer_G) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss_G.backward()

            model.module.optimizer_G.step()

            # update discriminator weights\n",
            model.module.optimizer_D.zero_grad()
            if config.fp16:
                with amp.scale_loss(loss_D, optimizer_D) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss_D.backward()
            model.module.optimizer_D.step()

            if (step + 1) % config.print_freq == 0 or step == total_steps - 1:
                logger.info(
                    "Train: [{:2d}/{}] Step {:03d}/{:03d}".format(
                        epoch + 1, config.epochs, i, len(data_set) - 1
                    )
                )
                logger.info(
                    "Loss D: {},  Loss G {}, Loss VGG {}".format(
                        loss_D.item(), loss_dict['G_GAN'].item(),
                        loss_dict['G_VGG'].item()
                    )
                )

            if save_gen:
                visuals = OrderedDict([('input_label',
                                        util.tensor2label(data['label'][0],
                                                          config.label_nc)),
                                       ('synthesized_image', util.tensor2im(
                                           generated.data[0])),
                                       ('real_image',
                                        util.tensor2im(data['image'][0]))])

                if not config.no_temporal_smoothing:
                    visuals['prev_label'] = util.tensor2label(prev_label[0],
                                                              config.label_nc)
                    visuals['prev_real'] = util.tensor2im(prev_real[0])
                    visuals['prev_generated'] = util.tensor2im(
                        prev_generated.data[0])

                visualizer.display_current_results(visuals, epoch, total_steps)

            if (step + 1) % config.save_latest_freq == 0 or \
                    step == total_steps - 1:
                model.module.save('latest')
                model.module.save(epoch)

            prev_generated = generated
            step += 1

        ### train the entire network after certain iterations
        if (config.niter_fix_global != 0) and (epoch == config.niter_fix_global):
            model.module.update_fixed_params()

        ### linearly decay learning rate after certain iterations
        if epoch > config.niter:
            model.module.update_learning_rate()


if __name__ == '__main__':
    train(*utils.start_run(TrainOptions))
