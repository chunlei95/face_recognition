import time

import torch
from tqdm import trange

from utils.logger import setup_logger


def evaluate(model,
             eval_loader,
             device,
             log_dir):
    logger = setup_logger(name='eval', output=log_dir)
    logger.info('start evaluating...')
    metric = dict()
    start = time.time()
    targets = None
    similarities = None
    for _ in trange(len(eval_loader)):
        for idx, (img_1, img_2, target) in enumerate(eval_loader):
            img_1 = img_1.to(device)
            img_2 = img_2.to(device)
            target = target.to(device).long()
            embed_1 = model(img_1)  # (b, embed_size)
            embed_2 = model(img_2)  # (b, embed_size
            similarity = calculate_similarity(embed_1, embed_2)
            if similarities is None:
                similarities = similarity
            else:
                similarities = torch.cat([similarities, similarity], dim=0)
            if targets is None:
                targets = target
            else:
                targets = torch.cat([targets, target], dim=0)
    end = time.time()
    acc, threshold = calculate_accuracy(similarities, targets)
    logger.info(
        f'[Eval] The evaluation accuracy is {acc} with threshold {threshold}, '
        f'time cost is {end - start}s, '
        f'avg per image cost is {((end - start) / len(similarities)) * 1000}ms, '
        f'FPS is {len(similarities) / (end - start)} images/s')
    logger.info('end evaluating....')
    metric['accuracy'] = acc
    return metric


def calculate_similarity(embed_1, embed_2):
    a_dot_b = torch.sum(embed_1 * embed_2, dim=1)
    m_a = torch.linalg.norm(embed_1, dim=1)
    m_b = torch.linalg.norm(embed_2, dim=1)
    similarity = a_dot_b / (m_a * m_b)
    return similarity


def calculate_accuracy(predict, target):
    """
    针对批量向量对的相似度计算并计算准确率
    :param predict: 预测的向量相似度
    :param target: (b,)，标签值，1表示相似，0表示不想似
    :return:
    """
    best_acc = 0.
    best_threshold = 0.
    for i in range(len(predict)):
        threshold = float(predict[i])
        similarity = torch.where(predict > threshold, 1., 0.)
        acc = torch.mean(torch.where(similarity == target, 1., 0.))
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold
    # auc = metrics.roc_auc_score(target, predict)
    # acc = metrics.accuracy_score(target, predict)
    return best_acc, best_threshold


if __name__ == '__main__':
    x1 = torch.tensor([[0, 1, 2], [2, 1, 1]], dtype=torch.float32)
    x2 = torch.tensor([[1, 2, 1], [1, 1, 0]], dtype=torch.float32)
    target = torch.tensor([1, 1], dtype=torch.long)
    s1 = torch.linalg.norm(x1, dim=1)
    s2 = torch.linalg.norm(x2, dim=1)
    print(s1)
    print(s2)
    print(torch.sum(x1 * x2, dim=1))
    similarity = torch.sum(x1 * x2, dim=1) / (s1 * s2)
    similarity = torch.where(similarity > 0.8, 1., 0.)
    acc = torch.mean(torch.where(target == similarity, 1., 0.))
    print(acc)
