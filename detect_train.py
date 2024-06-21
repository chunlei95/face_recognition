import argparse
import random

import numpy as np
import torch

from detection.core.train import train


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval',
                        dest='eval',
                        action='store_true',
                        default=True)
    parser.add_argument('--start_eval_iter',
                        dest='start_eval_iter',
                        type=int,
                        default=0)
    parser.add_argument('--eval_interval',
                        dest='eval_interval',
                        type=int,
                        default=1000)
    parser.add_argument('--seed',
                        dest='seed',
                        type=int,
                        default=1234)
    parser.add_argument('--deterministic',
                        dest='deterministic',
                        action='store_true',
                        default=True)
    parser.add_argument('--resume_model',
                        dest='resume_model',
                        type=str,
                        default=None)
    parser.add_argument('--start_iter',
                        dest='start_iter',
                        type=int,
                        default=0)
    parser.add_argument('--log_iter',
                        dest='log_iter',
                        type=int,
                        default=100)
    parser.add_argument('--log_dir',
                        dest='log_dir',
                        type=str,
                        default='logs/')
    parser.add_argument('--save_dir',
                        dest='save_dir',
                        type=str,
                        default='checkpoint/')
    return parser.parse_args()


def main(args):
    import detection.configs.yolo as cfg
    device = cfg.device
    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.random.manual_seed(args.seed)
        torch.cuda.random.manual_seed(args.seed)
        torch.cuda.random.manual_seed_all(args.seed)
    if args.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    model = cfg.model
    model.to(device)
    optimizer = cfg.optimizer
    lr_schedular = cfg.lr_schedular
    if args.start_iter != 0:
        if args.resume_model is None:
            raise RuntimeError('resume_model should be specified while start_iter is not 0.')
    if args.resume_model:
        # resume model checkpoint should contain the following parts
        checkpoint = torch.load(args.resume_model)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_schedular.load_state_dict(checkpoint['lr_schedular'])
        args.start_iter = checkpoint['iter']

    train(model=model,
          train_loader=cfg.train_loader,
          val_loader=cfg.val_loader,
          optimizer=optimizer,
          lr_schedular=lr_schedular,
          epoch=cfg.epoch,
          start_iter=args.start_iter,
          start_eval_iter=args.start_eval_iter,
          eval_interval=args.eval_interval,
          log_iter=args.log_iter,
          log_dir=args.log_dir,
          save_dir=args.save_dir,
          device=device)


if __name__ == '__main__':
    args = parse_args()
    main(args)
