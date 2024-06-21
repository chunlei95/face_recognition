import os

import torch

from detection.core.eval import evaluate
from utils.logger import setup_logger
from utils.loss import LossUtils


def train(model,
          train_loader,
          val_loader,
          optimizer,
          lr_schedular,
          epoch,
          start_iter,
          start_eval_iter,
          eval_interval,
          log_iter,
          log_dir,
          save_dir,
          device):
    logger = setup_logger('train', output=log_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    total_iters = len(train_loader) * epoch
    best_metric = 0.
    # cur_iter从1开始，而非从0开始，目的是为了方便后面的计算
    cur_iter = start_iter + 1 if start_iter == 0 else start_iter
    start_epoch = 0 if cur_iter <= len(train_loader) else cur_iter // len(train_loader) - 1
    model.train()
    loss_utils = LossUtils()
    for i in range(start_epoch, epoch):
        cur_epoch_iter = cur_iter % len(train_loader)
        logger.info('start training process...')
        for idx, data in enumerate(train_loader):
            loss = model(data)
            total_loss = loss['loss']
            loss_utils.update(total_loss.item())
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            lr_schedular.step()
            if cur_iter % log_iter == 0:
                logger.info(
                    f'[Epoch {i + 1}] ({cur_epoch_iter}/{len(train_loader)}): '
                    f'lr = {lr_schedular.get_last_lr()[-1]}, '
                    f'loss = {loss_utils.avg_loss}')
                loss_utils.reset()
            if (cur_iter >= start_eval_iter and cur_iter % eval_interval == 0) or (cur_iter >= total_iters):
                model.eval()
                with torch.no_grad():
                    cur_metric = evaluate(model, val_loader, log_dir)
                # update and save best model params
                if cur_metric >= best_metric:
                    logger.info(f'best metric from {best_metric} update to {cur_metric}')
                    best_metric = cur_metric
                    save_path_best = os.path.join(save_dir, 'best_model.pth')
                    torch.save(model, save_path_best)
                    logger.info(f'save best model checkpoint to {save_path_best}')
                # save last model checkpoint
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': model.state_dict(),
                    'lr_schedular': lr_schedular.state_dict(),
                    'iter': cur_iter + 1
                }
                save_path_last = os.path.join(save_dir, 'last_model.pth')
                torch.save(checkpoint, save_path_last)
                logger.info(f'save last model checkpoint to {save_path_last}')
                model.train()
            cur_epoch_iter += 1
            cur_iter += 1
        if cur_iter >= total_iters:
            logger.info(f'The training process is complete...')
            break
