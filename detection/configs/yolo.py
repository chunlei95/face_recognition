import torch.cuda
from torch.optim import SGD
from torch.optim.lr_scheduler import PolynomialLR
from torch.utils.data import DataLoader

from detection.datasets.WiderFace import WiderFaceDataset
from detection.datasets.transforms.transform import Compose, Resize, ToTensor, RandomFlip, Normalize
from detection.models.backbone.yolo_pafpn import YOLOPAFPN
from detection.models.head.yolo import YOLOV3Head
from detection.models.yolo import YOLO
from detection.utils.detect_utils import collect_dict_fn

batch_size = 16
epoch = 100

train_transforms = Compose(
    [
        Resize(target_size=[256, 256], keep_ratio=True),
        RandomFlip(direction='random'),
        ToTensor(),
        Normalize(mean=0.5, std=1.0, inplace=True)
    ]
)
val_transforms = Compose(
    [
        Resize(target_size=[256, 256], keep_ratio=True),
        RandomFlip(direction='random'),
        ToTensor(),
        Normalize(mean=0.5, std=1.0, inplace=True)
    ]
)

train_dataset = WiderFaceDataset(root_path='D:/dataset/WiderFace', img_path='WIDER_train/WIDER_train/images',
                                 annotation_path='retinaface_gt_v1.1/train/label.txt', transforms=train_transforms,
                                 mode='train')
train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, batch_size=4, collate_fn=collect_dict_fn)

val_dataset = WiderFaceDataset(root_path='D:/dataset/WiderFace', img_path='WIDER_val/WIDER_val/images',
                               annotation_path='retinaface_gt_v1.1/val/label.txt', transforms=val_transforms,
                               mode='val')
val_loader = DataLoader(val_dataset, shuffle=True, drop_last=True, batch_size=2, collate_fn=collect_dict_fn)

backbone = YOLOPAFPN()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
head = YOLOV3Head(in_channels=[256, 512, 1024],
                  num_class=1,
                  anchors=[[(10, 13), (16, 30), (33, 23)],
                           [(30, 61), (62, 45), (59, 119)],
                           [(116, 90), (156, 198), (373, 326)]],
                  strides=[8, 16, 32],
                  ignore_threshold=0.5,
                  device=device)
model = YOLO(backbone, head)

optimizer = SGD(
    params=model.parameters(),
    momentum=0.9,
    lr=0.01
)

lr_schedular = PolynomialLR(
    optimizer=optimizer,
    total_iters=epoch * len(train_loader)
)
