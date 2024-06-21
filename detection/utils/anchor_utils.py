import torch


def anchor_generator(base_anchors, feat_maps=None, strides=None, img_size=None):
    if feat_maps is None:
        assert strides is not None and img_size is not None
        W, H = img_size
        feat_maps = [(W / s, H / s) for s in strides]
    assert len(base_anchors) == len(feat_maps)
    all_anchors = []
    for anchors, feat_map in zip(base_anchors, feat_maps):
        w, h = feat_map
        anchors = torch.tensor(anchors)
        anchors = anchors.unsqueeze(1)
        anchors = anchors.repeat(1, h, 1)
        anchors = anchors.unsqueeze(2)
        anchors = anchors.repeat(1, 1, w, 1)  # (len(anchors), h, w, 2)
        all_anchors.append(anchors)
    return all_anchors


if __name__ == '__main__':
    base_anchors = [[(116, 90), (156, 198), (373, 326)],
                    [(30, 61), (62, 45), (59, 119)],
                    [(10, 13), (16, 30), (33, 23)]]
    all_anchors = anchor_generator(base_anchors, [[7, 7], [5, 5], [3, 3]])
    print(all_anchors[0])
