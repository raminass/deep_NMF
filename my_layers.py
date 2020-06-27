import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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


class FrNMFLayer(nn.Module):
    """
    Multiplicative update with Frobinus norm
    """

    def __init__(self, comp, features):
        super(FrNMFLayer, self).__init__()
        # an affine operation: y = Wx +b
        self.fc1 = nn.Linear(comp, comp, bias=False)
        self.fc2 = nn.Linear(features, comp, bias=False)

    def forward(self, y, x):
        denominator = self.fc1(y)
        numerator = self.fc2(x)
        denominator[denominator == 0] = EPSILON
        delta = torch.div(numerator, denominator)
        return torch.mul(delta, y)


class MultiFrDNMFNet(nn.Module):
    """
    Class for a DNMF with variable layers number.
    Input:
        -n_layers = number of layers to cinstruct the Net
        -comp = number of components for factorization
        -features = original features length for each sample vector(mutational sites)
    each layer is MU of Frobinus norm
    """

    def __init__(self, n_layers, comp, features):
        super(MultiFrDNMFNet, self).__init__()
        self.n_layers = n_layers
        self.deep_nmfs = nn.ModuleList(
            [FrNMFLayer(comp, features) for i in range(self.n_layers)]
        )

    def forward(self, h, x):
        # forward pass through the network
        for i, l in enumerate(self.deep_nmfs):
            h = l(h, x)
        return h


class SharedFrDNMFNet(nn.Module):
    """
    Class for a DNMF with variable layers number.
    Input:
        -n_layers = number of layers to cinstruct the Net
        -comp = number of components for factorization
        -features = original features length for each sample vector(mutational sites)
    each layer is MU of Frobinus norm
    """

    def __init__(self, n_layers, comp, features):
        super(SharedFrDNMFNet, self).__init__()
        self.n_layers = n_layers
        self.deep_nmf = FrNMFLayer(comp, features)

    def forward(self, h, x):
        # forward pass through the network
        for i in range(self.n_layers):
            h = self.deep_nmf(h, x)
        return h


class BetaNMFLayer(nn.Module):
    """
    mu for beta divergence based on Fevote article,
    beta=1 is KL
    beta=2 is Frobinus
    """

    def __init__(self, beta, comp, features):
        super(BetaNMFLayer, self).__init__()
        # an affine operation: y = Wx +b
        self.beta = beta
        self.fc1 = nn.Linear(comp, features, bias=False)  # WH
        self.fc2 = nn.Linear(features, comp, bias=False)  # W.t*

    # check if to use different fc2 layers
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
        super(UnsuperVisedDNMFNet, self).__init__()
        self.n_layers = n_layers
        self.deep_nmfs = nn.ModuleList(
            [BetaNMFLayer(beta, comp, features) for i in range(self.n_layers)]
        )

    def forward(self, h, x):
        # forward pass through the network
        for i, l in enumerate(self.deep_nmfs):
            h = l(h, x)
        return h


class UnsuperLayer(nn.Module):
    """
    Multiplicative update with Frobinus norm
    """

    def __init__(self, comp, features):
        super(UnsuperLayer, self).__init__()
        # an affine operation: y = Wx +b
        self.fc1 = nn.Linear(comp, comp, bias=False)
        self.fc2 = nn.Linear(features, comp, bias=False)
        self.fc1 = nn.Linear(features, features, bias=False)
        self.fc2 = nn.Linear(features, comp, bias=False)

    def forward(self, y, x):
        denominator = self.fc1(y)
        numerator = self.fc2(x)
        denominator[denominator == 0] = EPSILON
        delta = torch.div(numerator, denominator)
        return torch.mul(delta, y)


class UnsuperNet(nn.Module):
    """
    Class for a DNMF with variable layers number.
    Input:
        -n_layers = number of layers to cinstruct the Net
        -comp = number of components for factorization
        -features = original features length for each sample vector(mutational sites)
    each layer is MU of Frobinus norm
    """

    def __init__(self, n_layers, comp, features):
        super(UnsuperNet, self).__init__()
        self.n_layers = n_layers
        self.deep_nmfs = nn.ModuleList(
            [FrNMFLayer(comp, features) for i in range(self.n_layers)]
        )

    def forward(self, h, x):
        # forward pass through the network
        for i, l in enumerate(self.deep_nmfs):
            h = l(h, x)
        return h
