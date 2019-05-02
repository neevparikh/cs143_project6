import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import logging
import sys
import os

sys.path.append('src')
sys.path.append('src/pix2pixHD')
sys.path.append('src/pytorch-openpose')

def start_run(config_ctr):
    config_obj = config_ctr()
    config_obj.parse()
    config_obj.opt.path = os.path.join("log_dir", "name")
    config = config_obj.opt

    device = torch.device("cuda")

    # tensorboard
    writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
    writer.add_text('config', config.as_markdown(), 0)

    logger = get_logger(os.path.join(
        config.path, "{}.log".format(config.name)))
    config.print_params(logger.info)

    logger.info("Logger is set - training start")

    # set gpu device id
    logger.info("Set GPU device {}".format(config.gpu))
    torch.cuda.set_device(config.gpu)

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    torch.backends.cudnn.benchmark = True

    return config_obj, writer, logger

class CustomSchedule:
    def __init__(self, num_iters, param_values, intervals):
        assert len(param_values) == len(intervals) + 1

        sum_v = 0.
        self.totals = []
        self.slopes = []
        prev_val = param_values[0]
        for val, interv in zip(param_values[1:], intervals):
            self.totals.append(sum_v)
            interv *= num_iters
            sum_v += interv
            slope = (val - prev_val) / interv
            prev_val = val
            self.slopes.append(slope)

        assert abs(sum_v - num_iters) < 1e-6

        self.iter = 0
        self.values = param_values[:-1]
        self.num_iters = num_iters

    def calc(self):
        self.iter += 1
        if self.iter > self.num_iters:
            self.iter = self.num_iters
        for val, slope, total in zip(self.values[::-1], self.slopes[::-1], self.totals[::-1]):
            if self.iter > total:
                return val + slope * (self.iter - total)
        raise ValueError

def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger('darts')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger

class AverageMeter():
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
