import os.path

import torch
import torch.nn as nn

from core.eval import evaluate
from utils.logger import setup_logger
from utils.loss import LossUtils


def train(model,
          head,
          train_loader,
          val_loader,
          resume_model,
          optimizer,
          lr_schedular,
          epoch,
          start_eval_iter,
          eval_interval_iters,
          save_best,
          save_dir,
          log_iter,
          log_dir,
          device):
    cross_entropy = nn.CrossEntropyLoss()
    logger = setup_logger('train', output=log_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    validation = True if val_loader else False
    iters_epoch = len(train_loader)
    total_iters = iters_epoch * epoch
    start_epoch = 0
    cur_iter = 1
    best_metric = 0.
    loss_util = LossUtils()
    model.train()
    if resume_model is not None:
        resume = torch.load(resume_model)
        model.load_state_dict(resume['model'])
        optimizer.load_state_dict(resume['optimizer'])
        lr_schedular.load_state_dict(resume['lr_schedular'])
        start_epoch = resume['cur_epoch']
        cur_iter = resume['cur_iter']
    for i in range(start_epoch, epoch):
        cur_epoch_iter = cur_iter % len(train_loader)
        for idx, (image, target) in enumerate(train_loader):
            image = image.to(device)
            target = target.to(device).long()
            embeddings = model(image)
            output = head(embeddings, target)
            # loss = F.cross_entropy(output, target)
            loss = cross_entropy(output, target.squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_util.update(loss.item())
            # 记录日志
            if cur_iter % log_iter == 0:
                logger.info(
                    f'[Epoch {i + 1}][Iter {cur_epoch_iter}/{iters_epoch}]: lr={lr_schedular.get_last_lr()}, loss={loss_util.avg_loss}')
                loss_util.reset()
            # 验证流程
            if (cur_iter >= start_eval_iter and validation and cur_iter % eval_interval_iters == 0) or (
                    cur_iter >= total_iters):
                model.eval()
                with torch.no_grad():
                    cur_metric = evaluate(model, val_loader, device=device, log_dir=log_dir)
                if best_metric <= cur_metric['accuracy']:
                    best_metric = cur_metric['accuracy']
                    logger.info(f'the best evaluation metric update to {best_metric}')
                    if save_best:
                        save_path = os.path.join(save_dir, 'best_model.pth')
                        torch.save(model, save_path)
                        logger.info(f'save best model checkpoint to {save_path}')
                save_last_checkpoint(model, optimizer, lr_schedular, i + 1, cur_iter + 1, save_dir)
                logger.info(f'current best accuracy is {best_metric}')
                model.train()
            cur_epoch_iter += 1
            cur_iter += 1
            lr_schedular.step()
        if cur_iter >= total_iters:
            logger.info(f'The training process is complete...')
            break


def save_last_checkpoint(model, optimizer, lr_schedular, cur_epoch, cur_iter, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, 'last_checkpoint.pth')
    dic = dict(
        model=model,
        optimizer=optimizer,
        lr_schedular=lr_schedular,
        cur_epoch=cur_epoch,
        cur_iter=cur_iter
    )
    torch.save(dic, save_path)
