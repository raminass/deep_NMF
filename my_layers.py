import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPSILON = torch.finfo(torch.float32).eps


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


class MultiDNMFNet(nn.Module):
    """
    Class for a DNMF with variable layers number.
    Input:
        -n_layers = number of layers to cinstruct the Net
        -comp = number of components for factorization
        -features = original features length for each sample vector(mutational sites)

    """

    def __init__(self, n_layers, comp, features):
        super(MultiDNMFNet, self).__init__()
        self.n_layers = n_layers
        self.deep_nmfs = nn.ModuleList(
            [NMFLayer(comp, features) for i in range(self.n_layers)]
        )

    def forward(self, h, x):
        # forward pass through the network
        for i, l in enumerate(self.deep_nmfs):
            h = l(h, x)
        return h


class BetaNMFLayer(nn.Module):
    def __init__(self, beta, comp, features):
        super(BetaNMFLayer, self).__init__()
        # an affine operation: y = Wx +b
        self.beta = beta
        self.fc1 = nn.Linear(comp, features, bias=False)  # WH
        self.fc2 = nn.Linear(features, comp, bias=False)  # W.t*

    def forward(self, y, x):
        wh = self.fc1(y)
        denominator = self.fc2(wh.pow(self.beta - 1))
        numerator = self.fc2(torch.mul(x, wh.pow(self.beta - 2)))
        denominator[denominator == 0] = EPSILON
        delta = torch.div(numerator, denominator)
        return torch.mul(delta, y)


class MultiBetaDNMFNet(nn.Module):
    """
    Class for a DNMF with variable layers number.
    Input:
        -n_layers = number of layers to cinstruct the Net
        -beta = beta divergence
        -comp = number of components for factorization
        -features = original features length for each sample vector(mutational sites)
    """

    def __init__(self, n_layers, beta, comp, features):
        super(MultiBetaDNMFNet, self).__init__()
        self.n_layers = n_layers
        self.deep_nmfs = nn.ModuleList(
            [BetaNMFLayer(beta, comp, features) for i in range(self.n_layers)]
        )

    def forward(self, h, x):
        # forward pass through the network
        for i, l in enumerate(self.deep_nmfs):
            h = l(h, x)
        return h



class UnsuperVisedDNMFNet(nn.Module):
    """
    Class for a DNMF with variable layers number.
    Input:
        -n_layers = number of layers to cinstruct the Net
        -beta = beta divergence
        -comp = number of components for factorization
        -features = original features length for each sample vector(mutational sites)
    """

    def __init__(self, n_layers, beta, comp, features):
        super(MultiBetaDNMFNet, self).__init__()
        self.n_layers = n_layers
        self.deep_nmfs = nn.ModuleList(
            [BetaNMFLayer(beta, comp, features) for i in range(self.n_layers)]
        )

    def forward(self, h, x):
        # forward pass through the network
        for i, l in enumerate(self.deep_nmfs):
            h = l(h, x)
        return h