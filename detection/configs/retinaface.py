import torch.cuda
from torch.optim import SGD
from torch.optim.lr_scheduler import PolynomialLR
from torch.utils.data import DataLoader

from detection.datasets.WiderFace import WiderFaceDataset
from detection.datasets.transforms.transform import Compose, Resize, RandomFlip, ToTensor, Normalize
from detection.models.head.rertinaface_head import RetinaFaceHead
from detection.utils.detect_utils import collect_dict_fn
from models.resnet import resnet50

epoch = 100
batch_size = 2
lr = 0.01
num_anchors = 2
out_channels = (2 + 10 + 4) * num_anchors

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_transforms = Compose(
    [
        Resize(target_size=[640, 640], keep_ratio=True),
        RandomFlip(direction='random'),
        ToTensor(),
        Normalize(mean=0.5, std=1.0, inplace=True)
    ]
)
val_transforms = Compose(
    [
        Resize(target_size=[640, 640], keep_ratio=True),
        RandomFlip(direction='random'),
        ToTensor(),
        Normalize(mean=0.5, std=1.0, inplace=True)
    ]
)
train_dataset = WiderFaceDataset(root_path='', img_path='', annotation_path='', transforms=train_transforms)
val_dataset = WiderFaceDataset(root_path='', img_path='', annotation_path='', transforms=val_transforms)
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collect_dict_fn, shuffle=True,
                          drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=512, collate_fn=collect_dict_fn)

model = resnet50(
    num_classes=out_channels,
    pretrained=True
)

head = RetinaFaceHead(

)

optimizer = SGD(
    params=[{'params': model.parameters()},
            {'params': head.parameters()}],
    lr=lr,
    momentum=0.9
)

lr_schedular = PolynomialLR(
    optimizer=optimizer,
    total_iters=epoch * len(train_loader)
)
