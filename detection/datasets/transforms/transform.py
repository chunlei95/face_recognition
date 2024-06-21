import random
from typing import Dict, Union, List, Tuple

import cv2
import torch
import torchvision.transforms.functional as F
from numpy import ndarray
from torch import Tensor
import numpy as np


class Compose:
    def __init__(self, transforms: Union[List, Tuple]):
        super().__init__()
        self.transforms = transforms

    def __call__(self, data: Dict):
        for t in self.transforms:
            data = t(data)
        return data


class Resize:
    """
    target_size: int or Union[tuple((w, h)), list([w, h])],
    """

    def __init__(self, target_size, keep_ratio):
        super().__init__()
        if type(target_size) == int:
            self.target_size = [target_size] * 2
        elif type(target_size) in (list, tuple):
            self.target_size = target_size
        else:
            raise ValueError('parameter of target_size is invalid!')
        self.keep_ratio = keep_ratio

    def _apply_image(self, img) -> ndarray:
        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_CUBIC)
        return img

    def _apply_boxes(self, sw, sh, boxes: ndarray) -> ndarray:
        # xywh
        boxes[..., 0::2] = boxes[..., 0::2] * sw
        boxes[..., 1::2] = boxes[..., 1::2] * sh
        # boxes[..., 2::2] = boxes[..., 2::2] * sw
        # boxes[..., 3::2] = boxes[..., 3::2] * sh
        return boxes

    def _apply_landmarks(self, sw, sh, landmarks: ndarray) -> ndarray:
        landms = landmarks[..., 0:10]
        landms_valid = landmarks[..., 10:11]
        landms[..., 0::2] = landms[..., 0::2] * sw
        landms[..., 1::2] = landms[..., 1::2] * sh
        landmarks = np.concatenate([landms, landms_valid], axis=-1)
        return landmarks

    def __call__(self, data: Dict):
        img = data['img']  # img的形状为HWC
        img_h, img_w = img.shape[:2]
        # target_size的形状为HW
        s_h = self.target_size[0] / img_h
        s_w = self.target_size[1] / img_w
        data['ori_scale'] = [img_w, img_h]
        data['scale'] = self.target_size
        data['scale_factor'] = [s_w, s_h]
        data['img'] = self._apply_image(data['img'], )
        if data.get('gt_boxes', None) is not None:
            data['gt_boxes'] = self._apply_boxes(s_w, s_h, data['gt_boxes'])
        if data.get('gt_landmarks', None) is not None:
            data['gt_landmarks'] = self._apply_landmarks(s_w, s_h, data['gt_landmarks'])
        return data


class RandomDistort:
    def __init__(self):
        pass

    def __call__(self, data: Dict):
        pass


class Pad:
    def __init__(self, pad_size):
        self.pad_size = pad_size

    def __call__(self, data: Dict):
        pass


class RandomFlip:
    def __init__(self, direction: str):
        directions = ['vertical', 'horizontal', 'both']
        if direction not in ('vertical', 'horizontal', 'both', 'random'):  # both表示水平垂直反转
            raise ValueError('invalid parameter value of direction!')
        if direction == 'random':
            direction = directions[random.randint(0, len(directions) - 1)]
        self.direction = direction

    def _apply_image(self, img: ndarray):
        if self.direction == 'vertical':
            img = cv2.flip(img, 0)
        elif self.direction == 'horizontal':
            img = cv2.flip(img, 1)
        elif self.direction == 'both':
            img = cv2.flip(img, -1)
        return img

    def _apply_boxes(self, boxes: ndarray, img_size: Union[List, Tuple]) -> ndarray:
        W, H = img_size
        flipped = boxes.copy()
        # boxes的形式为xywh
        if self.direction == 'vertical':
            flipped[..., 1] = H - boxes[..., 3]
            # flipped[..., 3] = H - boxes[..., 1]
        elif self.direction == 'horizontal':
            flipped[..., 0] = W - boxes[..., 2]
            # flipped[..., 2] = W - boxes[..., 0]
        elif self.direction == 'both':
            flipped[..., 0] = W - boxes[..., 2]
            flipped[..., 1] = H - boxes[..., 3]
            # flipped[..., 2] = W - boxes[..., 0]
            # flipped[..., 3] = H - boxes[..., 1]
        return flipped

    def _apply_landmarks(self, landmarks: ndarray, img_size: Union[list, Tuple]) -> ndarray:
        W, H = img_size
        landm = landmarks[..., 0:10]
        landm_valid = landmarks[..., 10:]
        flipped = landm.copy()
        if self.direction == 'vertical':
            flipped[..., 1::2] = H - landm[..., 1::2]
        elif self.direction == 'horizontal':
            flipped[..., 0::2] = W - landm[..., 0::2]
        elif self.direction == 'both':
            flipped[..., 0::2] = W - landm[..., 0::2]
            flipped[..., 1::2] = H - landm[..., 1::2]
        flipped = np.concatenate([flipped, landm_valid], axis=-1)
        return flipped

    def __call__(self, data: Dict):
        data['flip'] = True
        data['flip_direction'] = self.direction
        data['img'] = self._apply_image(data['img'])  # (w, h, c)
        img_size = data['img'].shape[:2]  # (w, h)
        if data.get('gt_boxes', None) is not None:
            data['gt_boxes'] = self._apply_boxes(data['gt_boxes'], img_size)
        if data.get('gt_landmarks', None) is not None:
            data['gt_landmarks'] = self._apply_landmarks(data['gt_landmarks'], img_size)
        return data


class Normalize:
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def _apply_images(self, image: Tensor):
        return F.normalize(image, self.mean, self.std, self.inplace)

    # def _apply_boxes(self, boxes, img_size):
    #     H, W = img_size
    #     boxes[..., 0::2] /= W
    #     boxes[..., 1::2] /= H
    #     return boxes
    #
    # def _apply_landmarks(self, landmarks, img_size):
    #     H, W = img_size
    #     landmarks[..., 0::2] /= W
    #     landmarks[..., 1::2] /= H
    #     return landmarks

    def __call__(self, data: Dict):
        data['img'] = self._apply_images(data['img'])  # Tensor: (C, H, W)
        # img_size = data['img'].shape[1:]  # (H, W)
        # if data.get('gt_boxes', None) is not None:
        #     data['gt_boxes'] = self._apply_boxes(data['gt_boxes'], img_size)
        # if data.get('gt_landmarks', None) is not None:
        #     data['gt_landmarks'] = self._apply_landmarks(data['gt_landmarks'], img_size)
        return data


class ToTensor:
    def __init__(self):
        super().__init__()

    def __call__(self, data: Dict):
        img = data['img']
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img / 255.0)
        img = torch.permute(img, (2, 0, 1))
        data['img'] = img
        if data.get('gt_boxes', None) is not None:
            data['gt_boxes'] = torch.from_numpy(data['gt_boxes'])
        if data.get('gt_landmarks', None) is not None:
            data['gt_landmarks'] = torch.from_numpy(data['gt_landmarks'])
        if data.get('gt_conf', None) is not None:
            data['gt_conf'] = torch.from_numpy(data['gt_conf'])
        return data


class XYXY2XYWH:
    def __init__(self):
        super().__init__()

    def _apply_boxes(self, boxes: ndarray):
        pass

    def _apply_landmarks(self, landmarks: ndarray):
        pass

    def __call__(self, data: Dict):
        pass
