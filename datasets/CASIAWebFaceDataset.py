from glob import glob

import torch
from PIL import Image
import cv2
from torch.utils.data import Dataset


class CASIAWebFace(Dataset):
    def __init__(self, root_path, transforms=None):
        super().__init__()
        self.root_path = root_path
        img_paths = glob(self.root_path + '/*')
        self.image_list = []
        self.label_list = []
        for i in range(len(img_paths)):
            cls_imgs = glob(img_paths[i] + '/*')
            self.image_list.extend(cls_imgs)
            self.label_list.extend([i] * len(cls_imgs))
        assert len(self.image_list) == len(self.label_list)
        self.transforms = transforms

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        image = self.image_list[item]
        label = self.label_list[item]
        image = Image.open(image)
        # image = cv2.cvtColor(cv2.imread(image, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if self.transforms is not None:
            image = self.transforms(image)
        return image, torch.tensor(label, dtype=torch.float32)
