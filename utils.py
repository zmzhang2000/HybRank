import os
import math
from collections import defaultdict
import logging
import torch


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        if not math.isfinite(val):
            val = 10000.0
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        num = self.avg
        if abs(num) > 1e-4:
            return f"{num:.4f}"
        else:
            return f"{num:.4e}"


class Metrics:
    def __init__(self, delimiter=" "):
        self.meters = defaultdict(AverageMeter)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, list) or isinstance(v, tuple):
                v, n = v
            else:
                v, n = v, 1

            if isinstance(v, torch.Tensor):
                v = v.item()
            if not math.isfinite(v):
                v = 10000.0
            assert isinstance(v, (float, int))
            self.meters[k].update(v, n)

    def reset_meters(self):
        for k in self.meters:
            if isinstance(self.meters[k], AverageMeter):
                self.meters[k].reset()

    def __str__(self):
        meters_str = [name + ": " + str(meter) for name, meter in self.meters.items()]
        return self.delimiter.join(meters_str)


def setup_logger(logger, args, filename='log'):
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    log_formatter = logging.Formatter(fmt="[%(asctime)s][%(levelname)s] - %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(log_formatter)
    file_handler = logging.FileHandler(os.path.join(args.output_dir, filename))
    file_handler.setFormatter(log_formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
