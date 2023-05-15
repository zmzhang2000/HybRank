import argparse
import random
import logging
import torch
from torch import cuda
import numpy as np

logger = logging.getLogger()


def get_args():
    parser = argparse.ArgumentParser()
    # training experiment name
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--data_path', type=str, default=None)

    # model related params
    parser.add_argument('--num_sim_type', type=int, default=2)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=8)

    # training specific args
    parser.add_argument('--only_eval', action='store_true', default=False)
    parser.add_argument('--resume', type=str, default=None)

    parser.add_argument('--list_len', type=int, default=100)
    parser.add_argument('--num_anchors', type=int, default=100)

    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda' if cuda.is_available() else 'cpu')

    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-6)

    parser.add_argument('--max_grad_norm', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--log_per_steps', type=int, default=100)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return args


def print_args(args):
    logger.info(f"{'*' * 30} CONFIGURATION {'*' * 30}")
    for key, val in sorted(vars(args).items()):
        keystr = "{}".format(key) + (" " * (30 - len(key)))
        logger.info("%s -->   %s", keystr, val)
    logger.info(f"{'*' * 30} CONFIGURATION {'*' * 30}")
