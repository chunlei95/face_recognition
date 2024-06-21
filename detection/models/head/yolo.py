import torch
import torch.nn as nn
import torch.nn.functional as F

from detection.models.anchor.anchor_assigner import GridAssigner
from detection.utils.anchor_utils import anchor_generator
from detection.utils.detect_utils import make_grid, xywh2xyxy, calculate_batch_iou


class YOLOV3Head(nn.Module):
    def __init__(self, in_channels, num_class, anchors, strides, ignore_threshold, device):
        super().__init__()
        self.in_channels = in_channels
        self.anchors = anchors
        self.strides = strides
        out_channels = (4 + 1 + 10) * len(self.anchors)
        self.heads = nn.ModuleList([
            nn.Conv2d(in_channels[i], out_channels, 1) for i in range(len(in_channels))
        ])
        self.ignore_threshold = ignore_threshold
        self.anchors_assigner = GridAssigner(anchors, strides, num_class, device, bboxes_style='xywh')

    def loss(self, logits, targets):
        gt_boxes = targets['gt_boxes']
        gt_conf = targets['gt_conf']
        gt_landmarks = targets['gt_landmarks']
        gt_targets = self.anchors_assigner(logits, gt_boxes=gt_boxes, gt_landmarks=gt_landmarks,
                                           gt_conf=gt_conf)  # [(b, na, h, w, c), (), ()

        gt_targets = torch.cat([target.flatten(1, 3) for target in gt_targets], dim=1)
        gboxes = gt_targets[..., 0:4]  # (b, na, h, w, 4) -> (b, m1, 4)
        gx, gy = gboxes[..., 0:1], gboxes[..., 1:2]
        gw, gh = gboxes[..., 2:3], gboxes[..., 3:4]
        gconf = gt_targets[..., 4:5]  # (b, m1, 1)
        glandm = gt_targets[..., 5:15]  # (b, m1, 10)
        glandm_valid = gt_targets[..., 15:].repeat(1, 1, 10)  # landmarks是否有效的标志
        glandm_valid = glandm_valid.long()

        predicts = self.decode_output(logits)  # [(b, na, h, w, c), (), ()]
        predicts = torch.cat([predict.flatten(1, 3) for predict in predicts], dim=1)
        pbboxes = predicts[..., 0:4]

        logits = torch.cat([logit.flatten(1, 3) for logit in logits], dim=1)
        px, py = logits[..., 0:1], logits[..., 1:2]
        pw, ph = logits[..., 2:3], logits[..., 3:4]
        pconf = logits[..., 4:5]
        plandm = logits[..., 5:]

        loss_x = torch.abs(px - gx) * gconf
        loss_y = torch.abs(py - gy) * gconf
        loss_w = torch.abs(pw - gw) * gconf
        loss_h = torch.abs(ph - gh) * gconf

        loss_xy = loss_x + loss_y
        loss_wh = loss_w + loss_h
        loss_xy = loss_xy.sum([1, 2]).mean()
        loss_wh = loss_wh.sum([1, 2]).mean()

        p_boxes = xywh2xyxy(pbboxes)
        gt_boxes = xywh2xyxy(gt_boxes)

        batch_iou_similarity = calculate_batch_iou(p_boxes, gt_boxes)  # (b, m1, m2), m1表示anchor总数，m2表示gt总数
        max_iou, max_idx = torch.max(batch_iou_similarity, dim=-1)
        iou_mask = torch.tensor(max_iou <= self.ignore_threshold, dtype=p_boxes.dtype)

        loss_landmark = torch.abs(plandm - glandm) * gconf * glandm_valid
        loss_landmark = loss_landmark.sum([1, 2]).mean()

        obj_loss = F.binary_cross_entropy(pconf, gconf, reduction='none')
        loss_obj = obj_loss * gconf
        loss_obj = loss_obj.sum(-1).mean()
        loss_noobj = obj_loss * (1 - gconf)
        loss_noobj = loss_noobj.squeeze(-1) * iou_mask
        loss_noobj = loss_noobj.sum(-1).mean()

        total_loss = (loss_xy + loss_wh + loss_landmark) * 5 + loss_obj + 0.5 * loss_noobj
        loss = {
            'loss_xy': loss_xy * 5,
            'loss_wh': loss_wh * 5,
            'loss_landmark': loss_landmark * 5,
            'loss_obj': loss_obj + 0.5 * loss_noobj,
            'loss': total_loss
        }
        return loss

    def decode_output(self, logits):
        # decode操作不要影响原本的输出结果，需要注意
        feat_maps = [logit.shape[2:4] for logit in logits]  # h, w
        anchors = anchor_generator(base_anchors=self.anchors, feat_maps=feat_maps)  # [(na, h, w, 2), (...), (...)]
        decoded_outputs = []
        for i in range(len(logits)):
            anchor = anchors[i]
            grid_h, grid_w = feat_maps[i]
            grid = make_grid(grid_w, grid_h)  # (h, w, 2)
            grid = grid[None, None, :, :, :]  # (1, 1, h, w, 2)
            logit = logits[i]  # (b, na, h, w, c_)
            x, y = logit[..., 0:1], logit[..., 1:2]
            w, h = logit[..., 2:3], logit[..., 3:]
            # 计算出的x, y为在原图上的坐标
            x = (x + grid[..., 0:1]) * self.strides[i]
            y = (y + grid[..., 1:2]) * self.strides[i]
            # 计算出的w, h为在原图中的宽高
            w = (torch.exp(w) * anchor[..., 0:1])
            h = (torch.exp(h) * anchor[..., 1:2])
            out = torch.cat([x, y, w, h], dim=-1)
            decoded_outputs.append(out)
        return decoded_outputs

    def post_processing(self, logits):
        return logits

    def forward(self, inputs, labels):
        assert len(inputs) == len(self.in_channels)
        logits = []
        level_anchors = len(self.anchors[0])
        for inp, head in zip(inputs, self.heads):
            logit = head(inp)
            b, c, h, w = logit.shape
            logit = logit.permute((0, 2, 3, 1)).reshape((b, h, w, level_anchors, c // level_anchors))
            logit = logit.permute((0, 3, 1, 2, 4))  # (b, na, h, w, c_)
            pos_xy = logit[..., 0:2]
            pos_wh = logit[..., 2:4]
            conf = logit[..., 4:5]
            landm = logit[..., 5:]
            pos_xy = F.sigmoid(pos_xy)
            conf = F.sigmoid(conf)
            landm = F.sigmoid(landm)
            logit = torch.cat([pos_xy, pos_wh, conf, landm], dim=-1)
            # 没有进行decode的outputs，计算损失需要用
            logits.append(logit)
        if self.training:
            return self.loss(logits, labels)
        else:
            logits = self.post_processing(logits)
            return logits
