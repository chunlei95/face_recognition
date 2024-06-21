import argparse
import random

import numpy as np
import torch

import configs.config as cfg
from core.train import train


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval',
                        dest='eval',
                        action='store_true',
                        default=True,
                        help='whether to do evaluate during training')
    parser.add_argument('--start_eval_iter',
                        dest='start_eval_iter',
                        type=int,
                        default=0)
    parser.add_argument('--eval_interval_iters',
                        dest='eval_interval_iters',
                        type=int,
                        default=1000)
    parser.add_argument('--resume_model',
                        dest='resume_model',
                        type=str,
                        default=None)
    parser.add_argument('--epoch',
                        dest='epoch',
                        type=int,
                        default=100,
                        help='total train epoch')
    parser.add_argument('--lr',
                        dest='lr',
                        type=float,
                        default=0.01,
                        help='learning rate')
    parser.add_argument('--batch_size',
                        dest='batch_size',
                        type=int,
                        default=16)
    parser.add_argument('--seed',
                        dest='seed',
                        type=int,
                        default=1234)
    parser.add_argument('--log_iters',
                        dest='log_iters',
                        type=int,
                        default=100)
    parser.add_argument('--log_dir',
                        dest='log_dir',
                        type=str,
                        default='logs/train.log')
    parser.add_argument('--save_best',
                        dest='save_best',
                        type=bool,
                        default=True,
                        help='whether to save best validation checkpoint during training')
    parser.add_argument('--save_dir',
                        dest='save_dir',
                        type=str,
                        default='checkpoints/')
    return parser.parse_args()


def main(args):
    device = cfg.device
    if args.seed is not None:
        # 设置全局随机数种子，使用确定性算法
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.random.manual_seed(args.seed)
        torch.cuda.random.manual_seed(args.seed)
        torch.cuda.random.manual_seed_all(args.seed)
        # 使用确定性算法，可复现能力强**************************
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # *************************************************
    else:
        # 使用非确定性算法，提升训练速度，可复现能力差*************
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        # *************************************************

    # 训练数据集，使用widerface
    train_loader = cfg.train_loader
    # 验证数据集，使用widerface
    val_loader = None
    if args.eval:
        val_loader = cfg.val_loader
    # 优化器
    optimizer = cfg.optimizer
    lr_schedular = cfg.lr_schedular
    # 模型配置，目前使用resnet
    model = cfg.backbone
    model.to(device)
    head = cfg.head
    head.to(device)

    train(model,
          head,
          train_loader,
          val_loader=val_loader,
          resume_model=args.resume_model,
          optimizer=optimizer,
          lr_schedular=lr_schedular,
          epoch=args.epoch,
          start_eval_iter=args.start_eval_iter,
          eval_interval_iters=args.eval_interval_iters,
          save_best=args.save_best,
          save_dir=args.save_dir,
          log_iter=args.log_iters,
          log_dir=args.log_dir,
          device=device)


if __name__ == '__main__':
    args = parse_args()
    main(args)
