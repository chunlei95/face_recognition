import torch


def nms(boxes, scores, threshold):
    keep = []
    # order是排序后的索引，返回值是一个tuple（Tensor， LongTensor），因为要表示索引，因此需要使用第二个返回值
    _, order = torch.sort(scores, descending=True)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])
        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h
        over = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (over <= threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        # +1是应为计算iou时是从第二个框开始的（第一个框与其余框的iou，长度少了1）
        order = order[ids + 1]
    return torch.LongTensor(keep)


def calculate_batch_iou(bboxes1, bboxes2, eps=1e-8):
    # bboxes1形状为(b, m1, 4), bboxes2形状为(b, m2, 4)
    bboxes1 = bboxes1.unsqueeze(2)  # (b, m1, 4) -> (b, m1, 1, 4)
    bboxes2 = bboxes2.unsqueeze(1)  # (b, m2, 4) -> (b, 1, m2, 4)
    px1y1 = bboxes1[:, :, :, 0:2]
    px2y2 = bboxes1[:, :, :, 2:4]
    gx1y1 = bboxes2[:, :, :, 0:2]
    gx2y2 = bboxes2[:, :, :, 2:4]
    x1y1 = torch.maximum(px1y1, gx1y1)
    x2y2 = torch.minimum(px2y2, gx2y2)
    overlap = (x2y2 - x1y1).clip(0).prod(dim=-1)
    area1 = (px2y2 - px1y1).clip(0).prod(dim=-1)
    area2 = (gx2y2 - gx1y1).clip(0).prod(dim=-1)
    union = area1 + area2 - overlap + eps  # eps是为了防止分母为0
    return overlap / union


def xywh2xyxy(boxes):
    # boxes的形状为(b, box_num, 4)，其中4表示(x, y, w, h)
    if boxes is None or len(boxes) == 0:
        return None
    if len(boxes.shape) == 2:
        boxes = boxes[None, :, :]
    x1 = boxes[:, :, 0] - (boxes[:, :, 1] / 2)
    y1 = boxes[:, :, 1] - (boxes[:, :, 3] / 2)
    x2 = boxes[:, :, 0] + (boxes[:, :, 1] / 2)
    y2 = boxes[:, :, 1] + (boxes[:, :, 3] / 2)
    # 输出形状为(b, box_num, 4)，其中4表示(x1, y1, x2, y2),
    # 需要注意的是转换后的坐标有可能是负值，在需要用到的地方（比如计算iou）需要处理为0
    return torch.stack([x1, y1, x2, y2], dim=-1)


def xyxy2xywh(boxes):
    # boxes形状为(b, box_num, 4), 其中4表示(x1, y1, x2, y2)
    if boxes is None or len(boxes) == 0:
        return None
    if len(boxes.shape) == 2:
        boxes = boxes[None, :, :]  # batch为1的情况
    x = (boxes[:, :, 0] + boxes[:, :, 2]) / 2  # center_x = (x1 + x2) / 2
    y = (boxes[:, :, 1] + boxes[:, :, 3]) / 2  # center_y = (y1 + y2) / 2
    w = boxes[:, :, 2] - boxes[:, :, 0]  # w = (x2 - x1)
    h = boxes[:, :, 3] - boxes[:, :, 1]  # h = (y2 - y1)
    # 输出形状为(b, box_num, 4), 4表示(x, y, w, h)
    return torch.cat([x, y, w, h], dim=-1)


def post_processing(predict, threshold):
    scores = predict[..., :2]
    # 获取每个预测框是否有物体的类别
    has_target = torch.argmax(scores, dim=-1)
    # 筛选出属于有物体的框
    has_target_idx = (has_target == 1)
    predict = predict[has_target_idx]
    scores = predict[..., :2]
    boxes = predict[..., 2:6]
    landmarks = predict[..., 6:]
    boxes, landmarks = output_decode(boxes, landmarks)
    boxes = xywh2xyxy(boxes)
    predict = nms(boxes, scores, threshold)
    return predict


def output_decode(boxes, landmarks, anchors):
    pass


def plot_rectangle(image, bbox):
    pass


def collect_dict_fn(batch):
    """

    :param batch: list(dict) => [
                        {'img_url':...,
                        'scene_id':...,
                        'scene_name':...,
                        'gt_boxes': list,
                        'gt_landmarks': list,
                        'gt_class': list},
                        'gt_num': len(gt_boxes)
                        {...},
                        {...},
                        ...
                  ]
    :return:
    """
    batch_size = len(batch)
    C, H, W = batch[0]['img'].shape
    mode = batch[0]['mode']
    max_batch_gt_num = max([gt['gt_num'] for gt in batch])
    images = torch.zeros((batch_size, C, H, W))
    gt_boxes = torch.zeros((batch_size, max_batch_gt_num, 4))
    gt_landmarks = None
    if mode == 'train':
        gt_landmarks = torch.zeros((batch_size, max_batch_gt_num, 11))
    gt_conf = torch.zeros((batch_size, max_batch_gt_num, 1))
    for i in range(len(batch)):
        data = batch[i]
        gt_num = data['gt_num']
        images[i] = data['img']
        gt_boxes[i, 0:gt_num] = data['gt_boxes']
        if mode == 'train':
            gt_landmarks[i, 0:gt_num] = data['gt_landmarks']
        gt_conf[i, 0:gt_num] = data['gt_conf'].view(-1, 1)
    batch_data = {
        'images': images,
        'gt_boxes': gt_boxes,
        'gt_landmarks': gt_landmarks,
        'gt_conf': gt_conf
    }
    return batch_data


def multiscale_target_converter(gt_targets, anchors, input_resolution, down_sample_ratio):
    """
    将标签分配到对应尺度的anchor
    :param gt_targets: dict，包括置信度（'gt_conf'）、关键点（'gt_landmark'）、位置（'gt_box'）三个键值对
                       gt_targets: {'gt_box': ..., 'gt_landmark': ..., 'gt_conf': ...}
    :param anchors: list，每个list包含一个尺度的anchor宽高列表：
                    [
                        [[h1, w1],[h2, w2],[h3, w3]], # 第一个尺度的anchor
                        [...],  # 第二个尺度的anchor
                        [...]  # 第三个尺度的anchor
                    ]
    :param input_resolution: 输入图像的分辨率
    :param down_sample_ratio: 三个尺度各自的下采样率
    :return:
    """
    pass


def multiscale_bbox_converter(predicts, input_resolution, down_sample_ratio):
    """
    将预测得到的多尺度结果转化为相对于原始图像的结果
    :param predicts:
    :param input_resolution:
    :param down_sample_ratio:
    :return:
    """
    pass


def decode_output(predicts, anchors, strides):
    pass


def make_grid(w, h):
    grid_x = torch.arange(0, w).view(-1)
    grid_y = torch.arange(0, h).view(-1)
    x, y = torch.meshgrid(grid_y, grid_x)
    grid = torch.stack([x, y], dim=-1)
    return grid


if __name__ == '__main__':
    h, w = 7, 7
    print(make_grid(h, w).shape)
