class MetricUtil():
    def __init__(self):
        self.total = 0.  # 总的accuracy
        self.count = 0  # count的最终个数本质上等于val_loader的长度

    def reset(self):
        self.total = 0.
        self.count = 0

    def update(self, acc):
        # acc等于一个batch的平均值accuracy值
        self.total += acc
        self.count += 1

    @property
    def accuracy(self):
        # 整个验证集的平均accuracy
        return self.total / self.count
