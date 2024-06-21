import math

import torch

from detection.utils.detect_utils import xyxy2xywh


class GridAssigner:
    """
    anchor标签分配 -> 正负样本选择
    可以放在batch_collect_fn中，但考虑到此处需要使用self.responsible属性，放在模型是输出之后、计算损失之前比较方便
    但一定是放在正负样本选择之前
    """

    def __init__(self, anchors, strides, cls_num, device, dtype=torch.float32, bboxes_style='xyxy'):
        super().__init__()
        self.anchors = anchors  # (total_anchors, 4), 4表示anchor的坐标，形式为(x1, y1, x2, y2)
        self.strides = strides
        self.cls_num = cls_num
        self.device = device
        self.dtype = dtype
        self.bboxes_style = bboxes_style
        self.responsible = []

    def __call__(self, feat_maps, gt_boxes, gt_landmarks, gt_conf):
        self.reset()
        return self.assign(feat_maps, gt_boxes, gt_landmarks, gt_conf)

    def reset(self):
        self.responsible = []

    def assign_single_level(self, level_anchors, feat_map, stride, gt_boxes, gt_landm, gt_conf):
        # gt_boxes的形状为(b, ng, 4)，ng为num_gt的缩写，表示gt框的数量
        w_f, h_f = feat_map.shape[2], feat_map.shape[3]
        b = len(gt_boxes)  # b等于batch_size
        ng = gt_boxes.shape[1]  # ng表示单个图像中gt的个数，对于一批数据来说，统一为一个值（该批数据中gt个数的最大值）
        na = len(level_anchors)
        gt_target = torch.zeros((b, na, h_f, w_f, 16))  # 4个坐标，1个置信度，10个landmarks坐标，1个landmark有效标志
        responsible = torch.zeros((b, na, h_f, w_f, ng))  # 当前尺度anchor与gt的对应关系，即该anchor负责预测哪个gt
        for i in range(b):
            boxes = gt_boxes[i]  # batch中的一个图像中的gts
            landm = gt_landm[i][..., 0:10]
            landm_valid = gt_landm[i][..., 10:11]
            assert len(boxes) == ng
            for j in range(ng):  # j表示gt的index
                x = boxes[j][0]
                y = boxes[j][1]
                w = boxes[j][2]
                h = boxes[j][3]
                landm[j][0::2] = landm[j][0::2] / (w_f * stride)
                landm[j][1::2] = landm[j][1::2] / (h_f * stride)
                if w <= 0 or h <= 0:
                    continue
                # 计算x, y相对于该网格的位置，处于0到1之间
                c_x = x / stride
                c_y = y / stride
                # 确定x, y在当前feat_map上的中心点在哪个网格
                grid_x = math.floor(c_x)
                grid_y = math.floor(c_y)
                # 计算中心坐标相对于其所在网格的偏移量
                offset_x = c_x - grid_x
                offset_y = c_y - grid_y
                for k in range(na):
                    anchor = level_anchors[k]  # (w, h, 2)
                    # 计算w, h，根据yolov2的公式，计算出t_w和t_h
                    w_k = torch.log(w / anchor[0])
                    h_k = torch.log(h / anchor[1])
                    gt_target[i, k, grid_x, grid_y, 0:4] = torch.tensor([offset_x, offset_y, w_k, h_k])
                gt_target[i, :, grid_x, grid_y, 4] = gt_conf[i][j]
                gt_target[i, :, grid_x, grid_y, 5:15] = landm[j]
                gt_target[i, :, grid_x, grid_y, 15:] = landm_valid[j]
                responsible[i, :, grid_x, grid_y, j] = 1
        self.responsible.append(responsible)
        return gt_target

    def assign(self, feat_maps, gt_boxes, gt_landm, gt_conf):
        # gt_boxes的形状为(b, na_gt, 4)
        assert len(self.strides) == len(feat_maps)
        if self.bboxes_style == 'xyxy':
            gt_boxes = xyxy2xywh(gt_boxes)
        gt_targets = []
        for i in range(len(feat_maps)):
            gt_targets.append(
                self.assign_single_level(self.anchors[i], feat_maps[i], self.strides[i], gt_boxes, gt_landm, gt_conf))
        return gt_targets  # 一个列表，列表元素个数为多尺度的个数，每个尺度形状为(b, na, h, w, 5 + cls), 没有经过筛选


if __name__ == '__main__':
    x = torch.randn((1000, 50))
    y = x.new_full((50,), -1)
    print(y.shape)
    print(y.dtype)
