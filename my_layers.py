import torch
import torch.nn as nn

EPSILON = torch.finfo(torch.float32).eps


# ============================ Basic Net ===============================
class FrNMFLayer(nn.Module):
    """
    Multiplicative update with Frobenius norm
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
        -n_layers = number of layers to construct the Net
        -comp = number of components for factorization
        -features = original features length for each sample vector(mutational sites)
    each layer is MU of Frobenius norm
    """

    def __init__(self, n_layers, comp, features):
        super(MultiFrDNMFNet, self).__init__()
        self.n_layers = n_layers
        self.deep_nmfs = nn.ModuleList(
            [FrNMFLayer(comp, features) for i in range(self.n_layers)]
        )

    def forward(self, h, x):
        """
        Sequencing the layer to form a deep network, and run the input forward through the net
        :param h: initial exposures/coef
        :param x: original vector V
        :return: reduced representation of V to lower dimension
        """
        for i, l in enumerate(self.deep_nmfs):
            h = l(h, x)
        return h


# ================================ Regularized Net ======================================
class RegLayer(nn.Module):
    """
    Multiplicative update with Frobenius norm
    A bias is added to the denominator to reflect the regularization constrains
    This can fit L1, L2 regularization.
    """

    def __init__(self, comp, features, l_1, l_2):
        super(RegLayer, self).__init__()
        # an affine operation: y = Wx +b
        self.l_1 = l_1
        self.l_2 = l_2
        self.fc1 = nn.Linear(comp, comp, bias=False)
        self.fc2 = nn.Linear(features, comp, bias=False)

    def forward(self, y, x):
        denominator = torch.add(self.fc1(y), self.l_2 * y + EPSILON)
        numerator = torch.add(self.fc2(x), -self.l_1)
        delta = torch.div(numerator, denominator)
        return torch.mul(delta, y)


class RegNet(nn.Module):
    """
    Class for a Regularized DNMF with varying layers number.
    Input:
        -n_layers = number of layers to construct the Net
        -comp = number of components for factorization
        -features = original features length for each sample vector(mutational sites)
    each layer is MU of Frobenius norm
    """

    def __init__(self, n_layers, comp, features, l_1, l_2):
        super(RegNet, self).__init__()
        self.n_layers = n_layers
        self.deep_nmfs = nn.ModuleList(
            [RegLayer(comp, features, l_1, l_2) for i in range(self.n_layers)]
        )

    def forward(self, h, x):
        # sequencing the layers and forward pass through the network
        for i, l in enumerate(self.deep_nmfs):
            h = l(h, x)
        return h


# ================================ shared w  ========================

class SharedFrDNMFNet(nn.Module):
    """
    Class for a DNMF with variable layers number.
    This network stacks the same layer through the network, so the weights are shared between layers.
    Input:
        -n_layers = number of layers to construct the Net
        -comp = number of components for factorization
        -features = original features length for each sample vector(mutational sites)
    each layer is MU of Frobenius norm
    """

    def __init__(self, n_layers, comp, features):
        super(SharedFrDNMFNet, self).__init__()
        self.n_layers = n_layers
        self.deep_nmf = FrNMFLayer(comp, features)

    def forward(self, h, x):
        # forward pass through the network (same layer)
        for i in range(self.n_layers):
            h = self.deep_nmf(h, x)
        return h


# ============================== General Beta Net ===================================
class BetaNMFLayer(nn.Module):
    """
    mu for beta divergence based on Fevotte article,
    beta=1 is KL
    beta=2 is Frobenius
    """

    def __init__(self, beta, comp, features, regularized=False):
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
        -n_layers = number of layers to construct the Net
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

# ============================== Unsupervised Net =================================

#
# class UnSuperLayer(nn.Module):
#     """
#     Multiplicative update with Frobenius norm
#     """
#
#     def __init__(self, comp, features):
#         super(UnSuperLayer, self).__init__()
#         self.w = torch.nn.Parameter(data=torch.Tensor(comp, features), requires_grad=True)
#         self.w.data.uniform_(0, 1)
#
#     def forward(self, y, x):
#         denominator = y.mm(self.w.mm(torch.transpose(self.w, 0, 1)))
#         numerator = x.mm(torch.transpose(self.w, 0, 1))
#         denominator[denominator == 0] = EPSILON
#         delta = torch.div(numerator, denominator)
#         return torch.mul(delta, y)
#
#
# class UnsuperNet(nn.Module):
#     """
#     Class for a DNMF with variable layers number.
#     Input:
#         -n_layers = number of layers to construct the Net
#         -comp = number of components for factorization
#         -features = original features length for each sample vector(mutational sites)
#     each layer is MU of Frobenius norm
#     """
#
#     def __init__(self, n_layers, comp, features):
#         super(UnsuperNet, self).__init__()
#         self.n_layers = n_layers
#         self.deep_nmfs = nn.ModuleList(
#             [UnSuperLayer(comp, features) for i in range(self.n_layers)]
#         )
#
#     def forward(self, h, x):
#         # forward pass through the network
#         for i, l in enumerate(self.deep_nmfs):
#             h = l(h, x)
#         return h
