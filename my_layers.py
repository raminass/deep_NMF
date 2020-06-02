import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPSILON = torch.finfo(torch.float32).eps


class WeightClipper(object):
    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, "weight"):
            w = module.weight.data
            w = w.clamp(min=0)
            module.weight.data = w


class NMFLayer(nn.Module):
    def __init__(self, comp, features):
        super(NMFLayer, self).__init__()
        # an affine operation: y = Wx +b
        self.fc1 = nn.Linear(comp, comp, bias=False)
        self.fc2 = nn.Linear(features, comp, bias=False)

    def forward(self, y, x):
        denominator = self.fc1(y)
        numerator = self.fc2(x)
        denominator[denominator == 0] = EPSILON
        delta = torch.div(numerator, denominator)
        return torch.mul(delta, y)


class DeepNMFModel(nn.Module):
    def __init__(self, comp, features):
        """
        built manually due to
        https://github.com/pytorch/pytorch/issues/19808#issuecomment-487257761
        """
        super(DeepNMFModel, self).__init__()
        self.layer1 = NMFLayer(comp, features)
        self.layer2 = NMFLayer(comp, features)
        self.layer3 = NMFLayer(comp, features)
        self.layer4 = NMFLayer(comp, features)
        self.layer5 = NMFLayer(comp, features)

    def forward(self, h, x):
        h1 = self.layer1(h, x)
        h2 = self.layer2(h1, x)
        h3 = self.layer3(h2, x)
        h4 = self.layer4(h3, x)
        h5 = self.layer5(h4, x)
        return h5


class MuReluNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = MuReluLayer()
        self.layer2 = MuReluLayer()
        self.layer3 = MuReluLayer()
        self.layer4 = MuReluLayer()
        self.layer5 = MuReluLayer()

    def forward(self, h, v):
        h1 = self.layer1(h, v)
        h2 = self.layer2(h1, v)
        h3 = self.layer3(h2, v)
        h4 = self.layer4(h3, v)
        h5 = self.layer5(h4, v)
        return h5


class MuNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = MuLayer()
        self.layer2 = MuLayer()
        self.layer3 = MuLayer()
        self.layer4 = MuLayer()
        self.layer5 = MuLayer()

    def forward(self, h, v):
        h1 = self.layer1(h, v)
        h2 = self.layer2(h1, v)
        h3 = self.layer3(h2, v)
        h4 = self.layer4(h3, v)
        h5 = self.layer5(h4, v)
        return h5
