import os.path

import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader

from detection.datasets.transforms.transform import Compose, Resize, RandomFlip, ToTensor, Normalize


class WiderFaceDataset(Dataset):
    def __init__(self, root_path, img_path, annotation_path, transforms=None, mode='train'):
        self.transforms = transforms
        self.mode = mode
        self.imgs_path = []
        self.words = []
        img_path = os.path.join(root_path, img_path)
        annotation_path = os.path.join(root_path, annotation_path)
        f = open(annotation_path, 'r')
        lines = f.readlines()
        isFirst = True
        labels = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    self.words.append(labels_copy)
                    labels.clear()
                path = line[2:]
                # path = annotation_path.replace('label.txt', 'images/') + path
                path = os.path.join(str(img_path), path)
                self.imgs_path.append(path)
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)

        self.words.append(labels)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, item):
        img = cv2.imread(self.imgs_path[item])
        height, width, _ = img.shape
        labels = self.words[item]
        annotations = np.zeros((0, 16))
        if len(labels) == 0:
            return annotations
        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 16))
            # bbox
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            # annotation[0, 2] = label[0] + label[2]  # x2
            # annotation[0, 3] = label[1] + label[3]  # y2
            annotation[0, 2] = label[2]  # w
            annotation[0, 3] = label[3]  # h
            # 添加一个置信度，表示是人脸的可信程度
            annotation[0, 4] = 1
            if self.mode == 'train':
                # landmarks
                annotation[0, 5] = label[4]  # l0_x
                annotation[0, 6] = label[5]  # l0_y
                annotation[0, 7] = label[7]  # l1_x
                annotation[0, 8] = label[8]  # l1_y
                annotation[0, 9] = label[10]  # l2_x
                annotation[0, 10] = label[11]  # l2_y
                annotation[0, 11] = label[13]  # l3_x
                annotation[0, 12] = label[14]  # l3_y
                annotation[0, 13] = label[16]  # l4_x
                annotation[0, 14] = label[17]  # l4_y
                # landmarks是否正确
                if (annotation[0, 4] < 0):
                    annotation[0, 15] = -1  # 无效
                else:
                    annotation[0, 15] = 1  # 有效
            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        data = dict(
            mode=self.mode,
            img=img,
            gt_boxes=target[..., 0:4],
            gt_conf=target[..., 4:5],
            gt_landmarks=target[..., 5:] if self.mode == 'train' else None,
            gt_num=len(target)
        )
        if self.transforms is not None:
            data = self.transforms(data)
        return data


if __name__ == '__main__':
    a = [11, 12, 22]
    b = a
    print(b)
    b[1] = 11
    print(a)
    from detection.utils.detect_utils import collect_dict_fn

    train_transforms = Compose(
        [
            Resize(target_size=[256, 256], keep_ratio=True),
            RandomFlip(direction='random'),
            ToTensor(),
            Normalize(mean=0.5, std=1.0, inplace=True)
        ]
    )

    dataset = WiderFaceDataset(root_path='D:/dataset/WiderFace', img_path='WIDER_train/WIDER_train/images',
                               annotation_path='retinaface_gt_v1.1/train/label.txt', transforms=train_transforms)
    data_loader = DataLoader(dataset, shuffle=True, drop_last=True, batch_size=2, collate_fn=collect_dict_fn)
    for i, data in enumerate(data_loader):
        print(data)
    print('...')
