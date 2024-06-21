import torch.cuda
import torchvision.transforms as T
from torch.optim import SGD
from torch.optim.lr_scheduler import PolynomialLR
from torch.utils.data import DataLoader

from datasets.CASIAWebFaceDataset import CASIAWebFace
from datasets.LFWDataset import LFWDataset
from models.heads.arcface import ArcFaceHead
from models.resnet import resnet34

num_class = 10575
embed_size = 512
batch_size = 128
epoch = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'

backbone = resnet34(
    num_classes=embed_size,
    pretrained=True
)

head = ArcFaceHead(
    embed_size=512,
    num_classes=num_class,
    s=30,
    m=0.5,
    easy_margin=True,
    device=device
)

train_transforms = T.Compose([
    T.RandomCrop([112, 112]),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(0.5, 0.5)
])

val_transforms = T.Compose([
    T.CenterCrop([112, 112]),
    T.ToTensor(),
    T.Normalize(0.5, 0.5)
])

train_dataset = CASIAWebFace(
    # root_path='D:/dataset/WebFace/archive/casia-webface',
    root_path='/kaggle/input/casia-webface/casia-webface',
    transforms=train_transforms
)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True
)

val_dataset = LFWDataset(
    root_path='/kaggle/input/lfwtest/lfw-funneled',
    # root_path='D:/dataset/LFW/lfw-funneled',
    img_dir='lfw_funneled',
    pair_path='pairs.txt',
    # cached_path='',
    transforms=val_transforms
)
val_loader = DataLoader(
    val_dataset,
    batch_size=1024,
    shuffle=False,
    drop_last=False
)

optimizer = SGD(
    params=[{'params': backbone.parameters()}, {'params': head.parameters()}],
    momentum=0.9,
    lr=0.01
)

lr_schedular = PolynomialLR(
    optimizer=optimizer,
    total_iters=epoch * len(train_loader)
)
