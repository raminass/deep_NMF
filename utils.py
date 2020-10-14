import time
import numpy as np
import math
from dataclasses import dataclass
import torch
from my_layers import UnsuperNet,SuperNet
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import nnls
from sklearn.decomposition._nmf import _initialize_nmf as init_nmf

inf = math.inf
EPSILON = np.finfo(np.float32).eps


class WeightClipper(object):
    def __init__(self, lower=-inf, upper=inf):
        self.lower = lower
        self.upper = upper

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, "weight"):
            w = module.weight.data
            w = w.clamp(min=self.lower, max=self.upper)
            module.weight.data = w
  

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


def cost_function(v, w, h, l_1=0, l_2=0):
    d = v - np.dot(w, h)
    return (
        0.5 * np.power(d, 2).sum()
        + l_1 * np.abs(h).sum()
        + 0.5 * l_2 * np.power(h, 2).sum()
    )


def kl_reconstruct_error(x, w, h):
    a = x
    b = w.dot(h)
    return np.sum(
        np.where(a != 0, np.where(b != 0, a * np.log(a / b) - a + b, 0), 0)
    )  # from wp, zeros aren't counted
    # np.sqrt(2*np.sum(np.where(a != 0, np.where(b != 0, a * np.log(a / b) - a + b, 0), 0))) at Scikit, incorrect imo


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


def mu_update(V, W, H, l_1=0, l_2=0, update_H=True, update_W=True):
    # update W
    if update_W:
        W_nominator = np.dot(V, H.T)
        W_denominator = np.dot(W, np.dot(H, H.T)) + EPSILON
        delta = W_nominator / W_denominator
        W *= delta

    # update H
    if update_H:
        H_nominator = np.dot(W.T, V)
        H_denominator = np.dot(W.T.dot(W), H) + EPSILON + H * l_2 + l_1
        delta = H_nominator / H_denominator
        H *= delta
    return W, H


@dataclass
class Matensor:
    mat: np.ndarray
    tns: torch.Tensor


@dataclass
class Alldata:
    v_train: Matensor
    v_test: Matensor
    v_test: Matensor
    h_train: Matensor
    h_test: Matensor
    h_0_train: Matensor
    h_0_test: Matensor
    w: Matensor
    w_init: Matensor


def tensoring(X):
    # conver numpy array to torch tensor
    return torch.from_numpy(X).float()


def build_data(V, W, H, init_method="ones", TRAIN_SIZE=0.80):
    # create a dataclass object that includes all necessary datasets to train a model
    n_components = H.shape[0]
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
        h_train=Matensor(H[:, mask], tensoring(H[:, mask].T)),
        h_test=Matensor(H[:, ~mask], tensoring(H[:, ~mask].T)),
        h_0_train=Matensor(H_init[:, mask], tensoring(H_init[:, mask].T)),
        h_0_test=Matensor(H_init[:, ~mask], tensoring(H_init[:, ~mask].T)),
        w=Matensor(W, tensoring(W.T)),
        w_init=Matensor(W_init, tensoring(W_init.T)),
    )
    return data, n_components, features, samples

    
def train_supervised(
    data: Alldata,
    num_layers,
    network_train_iterations,
    L1,
    L2,
    verbose=False,
    lr=0.0008,
):
    v_train = data.v_train.tns
    h_train = data.h_train.tns
    h_0_train = data.h_0_train.tns

    n_components = h_train.shape[1]
    features = v_train.shape[1]
    # build the architicture
    deep_nmf = SuperNet(num_layers, n_components, features, L1, L2)
    for w in deep_nmf.parameters():
        w.data.fill_(1.0)
    criterion = nn.MSELoss(reduction="mean")
    test_cret = nn.MSELoss(reduction="mean")
    optimizerADAM = optim.Adam(deep_nmf.parameters(), lr=lr)
    
    # Train the Network
    inputs = (h_0_train, v_train)
    test_input = (data.h_0_test.tns, data.v_test.tns)
    loss_values = []
    test_loss = []
    for i in range(network_train_iterations):
        out = deep_nmf(*inputs)
        loss = criterion(out, h_train)  # loss between predicted and truth
        # loss = criterion(out.mm(W_tensor), v_train) # reconstruction loss

        if verbose:
            print(i, loss.item())

        optimizerADAM.zero_grad()
        loss.backward()
        optimizerADAM.step()

        # deep_nmf.apply(constraints)  # keep wieghts positive after gradient decent
        for w in deep_nmf.parameters():
            w.data = w.clamp(min=0,max=inf)
        loss_values.append(loss.item())

        # test performance
        test_out = deep_nmf(*test_input)
        test_loss.append(test_cret(test_out, data.h_test.tns).item())
    return deep_nmf, loss_values, test_loss


def train_unsupervised(
    data: Alldata,
    num_layers,
    network_train_iterations,
    n_components,
    verbose=False,
    lr=0.008,
    l_1=0,
    l_2=0,
    include_reg = True
):
    v_train = data.v_train.tns
    h_0_train = data.h_0_train.tns
    features = v_train.shape[1]
    # build the architicture
    if include_reg:
        deep_nmf = UnsuperNet(num_layers, n_components, features, l_1, l_2)
    else:
        deep_nmf = UnsuperNet(num_layers, n_components, features, 0, 0)
    # initialize parameters
    dnmf_w = data.w_init.tns
    for w in deep_nmf.parameters():
        w.data.fill_(0.1)

    optimizerADAM = optim.Adam(deep_nmf.parameters(), lr=lr)
     # Train the Network
    inputs = (h_0_train, v_train)
    test = (data.h_0_test.tns,data.v_test.tns)
    dnmf_train_cost = []
    dnmf_test_cost = []
    for i in range(network_train_iterations):
        out = deep_nmf(*inputs)
        test_out = deep_nmf(*test)

        R = v_train - out.mm(dnmf_w)
        loss = 0.5 * torch.sum(torch.mul(R, R)) + l_1 * out.sum() + 0.5 * l_2 * out.pow(2).sum()
        
        if verbose:
            print(i, loss.item())

        optimizerADAM.zero_grad()
        loss.backward()
        optimizerADAM.step()

        # keep weights positive after gradient decent
        for w in deep_nmf.parameters():
            w.data = w.clamp(min=0,max=inf)
        h_out = torch.transpose(out.data, 0, 1)
        h_out_t = out.data

        # NNLS
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html

        w_arrays = [nnls(out.data.numpy(), data.v_train.mat[f])[0] for f in range(features)]
        nnls_w = np.stack(w_arrays, axis=-1)
        dnmf_w = torch.from_numpy(nnls_w).float()
        
        dnmf_train_cost.append(loss.item())

        # test performance
        L = data.v_test.tns - test_out.mm(dnmf_w)
        test_loss = 0.5 * torch.sum(torch.mul(L, L)) + l_1 * test_out.sum() + 0.5 * l_2 * test_out.pow(2).sum()
        dnmf_test_cost.append(test_loss.item())
    return deep_nmf, dnmf_train_cost, dnmf_test_cost, dnmf_w
