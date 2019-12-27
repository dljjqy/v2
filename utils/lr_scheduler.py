class poly_lr(object):
    def __init__(self, basic_lr=0.01, power=0.9):
        self.basic_lr = basic_lr
        self.power = power

    def __call__(self, itr, max_iter):
        lr = pow((1-itr/max_iter), self.power)
        return lr