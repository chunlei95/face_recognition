import torch

from detection.utils.detect_utils import calculate_batch_iou
from detection.utils.detect_utils import xywh2xyxy


class YOLOMatcher:
    """
    YOLO的正负样本选择策略
    """

    def __init__(self, neg_iou_threshold, pos_iou_threshold):
        self.neg_iou_threshold = neg_iou_threshold
        self.pos_iou_threshold = pos_iou_threshold

    def __call__(self, p_boxes, gt_boxes, responsible):
        # responsible形状为(m1, m2)，表示哪个anchor负责哪个gt
        # p_boxes的形状均为(b, m1, 4)，m1表示所有尺度加在一起的anchor个数, 其中4表示(x, y, w, h)
        # gt_boxes的形状为(b, m2, 4), m2表示gt个数
        p_boxes = xywh2xyxy(p_boxes)
        gt_boxes = xywh2xyxy(gt_boxes)
        b = len(p_boxes)  # batch_size
        ta = p_boxes.shape[1]  # anchor总数，即m1
        batch_iou_similarity = calculate_batch_iou(p_boxes, gt_boxes)  # (b, m1, m2), m1表示anchor总数，m2表示gt总数
        # 返回值是一个形状为(b, m1)的tensor，元素为-1、0或1，分别表示忽略样本、负样本以及正样本
        sample_index = batch_iou_similarity.new_full((b, ta), -1)
        responsible = torch.cat([res.flatten(1, 3) for res in responsible], dim=1)
        # responsible = responsible[None, :, :]
        # 将anchor不负责预测的gt对应的iou清0
        # batch_iou_similarity = batch_iou_similarity * responsible
        # 首先选择每个gt的最好的预测框作为正样本，不管iou是多少
        max_iou, max_idx = torch.max(batch_iou_similarity, dim=-1)
        sample_index[:, max_idx] = 1
        # 清除已经选择的正样本
        batch_iou_similarity[..., max_idx] = -1
        # 找每个anchor预测出的最大iou（其实只有一个值，因为前面通过responsible过滤掉了与其它不相关gt的iou）
        max_iou, max_idx = torch.max(batch_iou_similarity, dim=0)
        neg_idx = max_idx[max_iou < self.neg_iou_threshold]
        sample_index[neg_idx] = 0
        pos_idx = max_idx[max_iou >= self.pos_iou_threshold]
        sample_index[pos_idx] = 1
        # 首先对预测框进行解码（xywh -> xyxy）
        # 计算所有预测框与所有真实框之间的iou
        # 初始化全部为忽略样本
        # 如果一个预测框与所有真实框之间的iou处于忽略阈值区间，设置为忽略样本
        return sample_index
