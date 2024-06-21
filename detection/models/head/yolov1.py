import torch.nn as nn


class YOLOV1Head(nn.Module):
    def __init__(self, conf_threshold, nms_threshold, num_classes, stride, device):
        super().__init__()

    def post_processing(self, conf_threshold, nms_threshold):
        pass

    def forward(self, inputs):
        pass
