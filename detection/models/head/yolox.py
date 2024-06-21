import torch.nn as nn


class YOLOXHead(nn.Module):
    def __init__(self, in_channels, num_class):
        super().__init__()
        self.in_channels = in_channels
        pos_out_channels = 4
        cls_out_channels = num_class
        self.pos_heads = nn.ModuleList([
            nn.Conv2d(in_channels[i], pos_out_channels, 1) for i in range(len(self.in_channels))
        ])
        self.cls_heads = nn.ModuleList([
            nn.Conv2d(in_channels[i], cls_out_channels, 1) for i in range(len(self.in_channels))
        ])

    def loss(self, pos_logits, conf_logits, cls_logits, targets):
        pass

    def forward(self, inputs, targets):
        assert len(inputs) == len(self.in_channels)
        pos_logits = []
        conf_logits = []
        cls_logits = []
        for i in range(len(inputs)):
            pos_logits.append(self.pos_heads[i](inputs[i]))
            cls_logits.append(self.cls_heads[i](inputs[i]))
        if self.training:
            return self.loss(pos_logits, conf_logits, cls_logits, targets)
        else:
            return pos_logits, conf_logits, cls_logits
