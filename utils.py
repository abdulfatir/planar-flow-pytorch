import torch

class Binarize(object):
    def __init__(self, threshold=0.5):
        self.threshold = threshold
    def __call__(self, tensor):
        return (tensor > self.threshold).type(tensor.type())
    def __repr__(self):
        return self.__class__.__name__ + '()'