import torch.nn as nn


class YOLO(nn.Module):
    def __init__(self, backbone, head, neck=None):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, inputs):
        x = inputs['images']
        outs = self.backbone(x)
        if self.neck is not None:
            outs = self.neck(outs)
        outs = self.head(outs, inputs)
        return outs
