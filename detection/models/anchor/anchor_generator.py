import torch
from torch.nn.modules.utils import _pair


class AnchorGenerator:
    def __init__(self, anchor_size, strides):
        super().__init__()

    def generate_base_anchor(self):
        pass

    def generate_base_anchor_single_level(self):
        pass

    def _mark_grid(self, shift_x, shift_y, row_major=True):
        xx = shift_x.repeat(shift_y.shape[0])
        yy = shift_y.view(-1, 1).repeat(1, shift_x.shape[0]).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def generate_grid_anchor(self, feat_maps):
        all_anchors = []
        for idx, feat in enumerate(feat_maps):
            all_anchors.append(self.generate_grid_anchor_single_level(idx, feat))
        return all_anchors

    def generate_grid_anchor_single_level(self, index, level_feat_map):
        base_anchor = self.base_anchors[index]
        stride = self.strides[index]
        stride_w, stride_h = stride
        lw, lh = level_feat_map
        shift_x = torch.arange(0, lw) * stride_w
        shift_y = torch.arange(0, lh) * stride_h
        shift_xx, shift_yy = self._mark_grid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        all_anchors = base_anchor[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)
        return all_anchors


class SSDAnchorGenerator(AnchorGenerator):
    pass


# noinspection PyMissingConstructor
class YOLOAnchorGenerator(AnchorGenerator):
    def __init__(self, anchor_size, strides):
        strides = [_pair(stride) for stride in strides]
        self.strides = strides
        self.centers = [(stride[0] / 2., stride[1] / 2.) for stride in self.strides]
        self.base_sizes = []
        for level_anchors in anchor_size:
            self.base_sizes.append([_pair(size) for size in level_anchors])
        self.base_anchors = self.generate_base_anchor(self.base_sizes, self.centers)

    def generate_base_anchor(self, base_sizes, centers):
        base_anchors = []
        for level_size, level_center in zip(base_sizes, centers):
            base_anchors.append(self.generate_base_anchor_single_level(level_size, level_center))
        return base_anchors

    def generate_base_anchor_single_level(self, level_size, level_center):
        anchors = []
        center_x, center_y = level_center
        for size in level_size:
            w, h = size
            anchors.append((center_x - w * 0.5, center_y - h * 0.5, center_x + w * 0.5, center_y + h * 0.5))
        return anchors


if __name__ == '__main__':
    generator = YOLOAnchorGenerator(anchor_size=[[(116, 90), (156, 198), (373, 326)],
                                                 [(30, 61), (62, 45), (59, 119)],
                                                 [(10, 13), (16, 30), (33, 23)]],
                                    strides=[32, 16, 8])
    print(generator)
