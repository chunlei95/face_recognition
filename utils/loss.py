class LossUtils():
    def __init__(self):
        super().__init__()
        self.total_loss = 0.
        self.iters = 0

    def reset(self):
        self.total_loss = 0.
        self.iters = 0

    def update(self, loss_value):
        self.total_loss += loss_value
        self.iters += 1

    @property
    def avg_loss(self):
        return self.total_loss / self.iters
