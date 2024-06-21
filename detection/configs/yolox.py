from detection.models.backbone.yolo_pafpn import YOLOPAFPN
from detection.models.head.yolox import YOLOXHead
from detection.models.yolox import YOLOX

backbone = YOLOPAFPN()

head = YOLOXHead(in_channels=[256, 512, 1024], num_class=1)

model = YOLOX(backbone=backbone, head=head)
