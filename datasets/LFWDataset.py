import os.path
from glob import glob

import torch
from PIL import Image
from torch.utils.data import Dataset


class LFWDataset(Dataset):
    def __init__(self, root_path, img_dir, pair_path, cached_path='cached_path', transforms=None):
        super().__init__()
        self.transforms = transforms
        self.img_list_1 = []
        self.img_list_2 = []
        self.ann_list = []
        # self.cached_img_path = os.path.join(self.cached_path, 'image_pair.npy')
        # self.cached_ann_path = os.path.join(self.cached_path, 'label.npy')
        # if os.path.exists(self.cached_img_path) and os.path.exists(self.cached_ann_path):
        #     self.img_list = np.load(self.cached_img_path)
        #     self.ann_list = np.load(self.cached_ann_path)
        # else:
        img_dir = os.path.join(root_path, img_dir)
        pair_path = os.path.join(root_path, pair_path)
        with open(pair_path) as f:
            lines = f.readlines()
            for line in lines:
                # while line is not None and line != '':
                strs = line.split()
                if len(strs) < 3:
                    continue
                # 图像对表示的是一个人
                elif len(strs) == 3:
                    img_path = os.path.join(str(img_dir), strs[0])
                    imgs = glob(img_path + '/*')
                    img_1 = imgs[int(strs[1]) - 1]
                    img_2 = imgs[int(strs[2]) - 1]
                    self.ann_list.append(1)  # 两张图片是同一个人，标签设置为1
                # 图像对中是不同的人
                else:
                    assert len(strs) == 4
                    img_path_1 = os.path.join(str(img_dir), strs[0])
                    img_path_2 = os.path.join(str(img_dir), strs[2])
                    imgs_1 = glob(img_path_1 + '/*')
                    imgs_2 = glob(img_path_2 + '/*')
                    img_1 = imgs_1[int(strs[1]) - 1]
                    img_2 = imgs_2[int(strs[3]) - 1]
                    self.ann_list.append(0)  # 两张图片不是同一个人，标签设置为0
                # 调整通道顺序，opencv读取的为BGR（WHC），转为RGB（CHW）

                # LFW数据集中为彩色图像
                # assert len(img_1.shape) == len(img_2.shape) == 3
                # 通道维度连接，前三个通道表示第一张图像，后三个通道表示第二张图像
                # pair = np.concatenate([img_1, img_2], axis=0)
                self.img_list_1.append(img_1)
                self.img_list_2.append(img_2)

    #     self.img_list = np.array(self.img_list)
    #     self.ann_list = np.array(self.ann_list)
    #     np.save(self.cached_img_path, self.img_list)
    #     np.save(self.cached_ann_path, self.ann_list)

    def __len__(self):
        return len(self.img_list_1)

    def __getitem__(self, item):
        img_1 = self.img_list_1[item]
        img_2 = self.img_list_2[item]
        img_1 = Image.open(img_1)
        img_2 = Image.open(img_2)
        # img_1 = cv2.cvtColor(cv2.imread(img_1, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        # img_2 = cv2.cvtColor(cv2.imread(img_2, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        ann = self.ann_list[item]
        if self.transforms is not None:
            img_1 = self.transforms(img_1)
            img_2 = self.transforms(img_2)
        return img_1, img_2, torch.tensor(ann, dtype=torch.float32)
