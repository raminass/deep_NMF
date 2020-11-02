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
        denominator = torch.add(self.fc1(y), self.l_2 * y + self.l_1 + EPSILON)
        numerator = self.fc2(x)
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

    def __init__(self, n_layers, comp, features, l_1=0, l_2=0):
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


# ================================ General DNMF Net ======================================
class DNMFLayer(nn.Module):
    """
    Multiplicative update with Frobenius norm
    This can fit L1, L2 regularization.
    """

    def __init__(self, comp, features, l_1, regularized):
        super(DNMFLayer, self).__init__()
        # an affine operation: y = Wx +b
        self.l_1 = l_1
        self.fc1 = nn.Linear(comp, comp, bias=regularized)
        self.fc2 = nn.Linear(features, comp, bias=False)

    def forward(self, y, x):
        denominator = torch.add(self.fc1(y), self.l_1 + EPSILON)
        numerator = self.fc2(x)
        delta = torch.div(numerator, denominator)
        return torch.mul(delta, y)


class DNMFNet(nn.Module):
    """
    Class for a Regularized DNMF with varying layers number.
    Input:
        -n_layers = number of layers to construct the Net
        -comp = number of components for factorization
        -features = original features length for each sample vector(mutational sites)
    each layer is MU of Frobenius norm
    """

    def __init__(self, n_layers, comp, features, l_1=0, regularized=False):
        super(DNMFNet, self).__init__()
        self.n_layers = n_layers
        self.deep_nmfs = nn.ModuleList(
            [RegLayer(comp, features, l_1, regularized) for i in range(self.n_layers)]
        )

    def forward(self, h, x):
        # sequencing the layers and forward pass through the network
        for i, l in enumerate(self.deep_nmfs):
            h = l(h, x)
        return h


# ================================ supervised Net ======================================
class SuperLayer(nn.Module):
    """
    Multiplicative update with Frobenius norm
    This can fit L1, L2 regularization.
    """

    def __init__(self, comp, features, L1, L2):
        super(SuperLayer, self).__init__()
        self.l_1 = L1
        self.l_2 = L2
        # an affine operation: y = Wx +b
        self.fc1 = nn.Linear(comp, comp, bias=False)
        self.fc2 = nn.Linear(features, comp, bias=False)

    def forward(self, y, x):
        denominator = torch.add(self.fc1(y), self.l_2 * y + self.l_1 + EPSILON)
        numerator = self.fc2(x)
        delta = torch.div(numerator, denominator)
        return torch.mul(delta, y)


class SuperNet(nn.Module):
    """
    Class for a Regularized DNMF with varying layers number.
    Input:
        -n_layers = number of layers to construct the Net
        -comp = number of components for factorization
        -features = original features length for each sample vector(mutational sites)
    each layer is MU of Frobenius norm
    """

    def __init__(self, n_layers, comp, features):
        super(SuperNet, self).__init__()
        L1 = nn.Parameter(torch.rand(1), requires_grad=True)
        L2 = nn.Parameter(torch.rand(1), requires_grad=True)
        self.n_layers = n_layers
        self.deep_nmfs = nn.ModuleList(
            [SuperLayer(comp, features, L1, L2) for i in range(self.n_layers)]
        )

    def forward(self, h, x):
        # sequencing the layers and forward pass through the network
        for i, l in enumerate(self.deep_nmfs):
            h = l(h, x)
        return h


# ================================ Unsupervised Net ======================================
class UnsuperLayer(nn.Module):
    """
    Multiplicative update with Frobenius norm
    This can fit L1, L2 regularization.
    """

    def __init__(self, comp, features, l_1, l_2):
        super(UnsuperLayer, self).__init__()
        # an affine operation: y = Wx +b
        self.l_1 = l_1
        self.l_2 = l_2
        self.fc1 = nn.Linear(comp, comp, bias=False)
        self.fc2 = nn.Linear(features, comp, bias=False)

    def forward(self, y, x):
        denominator = torch.add(self.fc1(y), self.l_2 * y + self.l_1 + EPSILON)
        numerator = self.fc2(x)
        delta = torch.div(numerator, denominator)
        return torch.mul(delta, y)


class UnsuperNet(nn.Module):
    """
    Class for a Regularized DNMF with varying layers number.
    Input:
        -n_layers = number of layers to construct the Net
        -comp = number of components for factorization
        -features = original features length for each sample vector(mutational sites)
    each layer is MU of Frobenius norm
    """

    def __init__(self, n_layers, comp, features, l_1=0, l_2=0):
        super(UnsuperNet, self).__init__()
        self.n_layers = n_layers
        self.deep_nmfs = nn.ModuleList(
            [UnsuperLayer(comp, features, l_1, l_2) for i in range(self.n_layers)]
        )

    def forward(self, h, x):
        # sequencing the layers and forward pass through the network
        for i, l in enumerate(self.deep_nmfs):
            h = l(h, x)
        return h


def initialize_exposures(V, n_components, method="random", seed=1984):
    """
    Average :
    this is initialization of w matrix as in scikiit transform method
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html#sklearn.decomposition.NMF.transform
    """
    n_features, n_samples = V.shape
    avg = np.sqrt(V.mean() / n_components)
    rng = np.random.RandomState(seed)

    if method == "random":
        exposures = avg * rng.rand(n_components, n_samples)
    elif method == "ones":
        exposures = np.ones((n_components, n_samples))
    elif method == "average":
        exposures = np.full((n_components, n_samples), avg)
    return exposures


def sBCD_update(V, W, H, O, obj="kl"):
    """
    Fast beta-divergence update, using: Fast Bregman Divergence NMF using Taylor Expansion and Coordinate Descent, Li 2012
    O parameter is a masking matrix, for cross-validation purposes, pass O=1
    """
    n, m = V.shape
    K = W.shape[1]
    V_tag = np.dot(W, H)
    E = np.subtract(V, V_tag)

    if obj == "kl":
        B = np.divide(1, V_tag) * O
    elif obj == "euc":
        B = np.ones((V.shape)) * O
    else:  # obj == 'is' Itakura-Saito
        B = np.divide(1, V_tag ** 2) * O

    for k in range(K):
        V_k = np.add(E, np.dot(W[:, k].reshape((n, 1)), H[k, :].reshape((1, m))))
        B_V_k = B * V_k
        # update H
        H[k] = np.maximum(
            1e-16, (np.dot(B_V_k.T, W[:, k])) / (np.dot(B.T, W[:, k] ** 2))
        )
        # update W
        W[:, k] = np.maximum(
            1e-16, (np.dot(B_V_k, H[k])) / (W[:, k] + np.dot(B, H[k] ** 2))
        )
        E = np.subtract(V_k, np.dot(W[:, k].reshape((n, 1)), H[k, :].reshape((1, m))))

    return W, H


def build_data_unsupervised(V, sigs, init_method="ones", TRAIN_SIZE=0.80):
    # create a dataclass object that includes all necessary datasets to train a model
    n_components = sigs
    features, samples = V.shape
    # split train/test
    TRAIN_SIZE = 0.80
    mask = np.random.rand(samples) < TRAIN_SIZE

    if init_method != "ones":
        W_init, H_init = init_nmf(V, n_components, init=init_method)
    else:
        H_init = np.ones((n_components, samples))
        W_init = np.ones((features, n_components))

    data = Alldata(
        v_train=Matensor(V[:, mask], tensoring(V[:, mask].T)),
        v_test=Matensor(V[:, ~mask], tensoring(V[:, ~mask].T)),
        h_train=Matensor(H_init[:, mask], tensoring(H_init[:, mask].T)),
        h_test=Matensor(H_init[:, ~mask], tensoring(H_init[:, ~mask].T)),
        h_0_train=Matensor(H_init[:, mask], tensoring(H_init[:, mask].T)),
        h_0_test=Matensor(H_init[:, ~mask], tensoring(H_init[:, ~mask].T)),
        w=Matensor(W_init, tensoring(W_init.T)),
        w_init=Matensor(W_init, tensoring(W_init.T)),
    )
    return data, n_components, features, samples


# plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots

# fig = plt.figure()
# ax = fig.add_subplot(111)
# bp_dict = ax.boxplot(np.log(df1[['super_reg','super_no_reg']].values),labels=['Regularized','Not-Regularized'],vert=True)

# for line in bp_dict['medians']:
#     # get position data for median line
#     x, y = line.get_xydata()[1] # top of median line
#     # overlay median value
#     text(x, y, '%.1f' % y,
#          verticalalignment='center') # draw above, centered
# ax.set_title('Supervised')
# ax.set_xlabel('Variant')
# ax.set_ylabel('$\log({MSE})$')
# plt.savefig('plots/figures/compare_reg_super.pdf')
# plt.show()