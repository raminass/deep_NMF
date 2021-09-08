import time
import numpy as np
import pandas as pd
import math
from dataclasses import dataclass
import torch
from ongoing_my_layers import *
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import nnls
from sklearn.decomposition._nmf import _initialize_nmf as init_nmf
from matplotlib import pyplot as plt
from matplotlib.pyplot import text
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


def cost_mat(v, w, h, l_1=0, l_2=0):
    # util.cost_mat(data.v_train.mat,data.w.mat,data.h_train.mat)
    d = v - np.dot(w, h)
    return (
        0.5 * np.power(d, 2).sum()
        + l_1 * h.sum()
        + 0.5 * l_2 * np.power(h, 2).sum()
    )/h.shape[1]


def cost_tns(v, w, h, l_1=0, l_2=0):
    # util.cost_tns(data.v_train.tns,data.w.tns,data.h_train.tns)
    d = v - h.mm(w)
    return (
        0.5 * torch.pow(d, 2).sum()
        + l_1 * h.sum()
        + 0.5 * l_2 * torch.pow(h, 2).sum()
    )/h.shape[0]


def kl_reconstruct_error(x, w, h, l_1=0, l_2=0):
    a = x + EPSILON
    b = w.dot(h) + EPSILON
    # from wp, zeros aren't counted
    kl = np.sum(a * np.log(a / b) - a + b) 
    return kl +  l_1 * h.sum() + 0.5 * l_2 * np.power(h, 2).sum()
    # np.sqrt(2*np.sum(np.where(a != 0, np.where(b != 0, a * np.log(a / b) - a + b, 0), 0))) at Scikit, incorrect imo

def kl_reconstruct_error_tns(x, w, h, l_1=0, l_2=0):
    a = x + torch.finfo(torch.float32).eps
    b = h.mm(w) + torch.finfo(torch.float32).eps
    kl = torch.sum(a * torch.log(a / b) - a + b) 
    return kl + l_1 * h.sum() + 0.5 * l_2 * torch.pow(h, 2).sum()

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

def mu_update_kl(V, W, H, l_1=0, l_2=0, update_H=True, update_W=True):
    # update W
    if update_W:
        W_nominator = np.dot(V / (W.dot(H) + EPSILON), H.T)
        W_denominator = np.dot(np.ones(V.shape), H.T) + EPSILON
        delta = W_nominator / W_denominator
        W *= delta

    # update H
    if update_H:
        H_nominator = np.dot(W.T, V / (W.dot(H) + EPSILON))
        H_denominator = np.dot(W.T, np.ones(V.shape)) + EPSILON + H * l_2 + l_1
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


def build_data(V, W, H, TRAIN_SIZE=0.80, index=None):
    # create a dataclass object that includes all necessary datasets to train a model
    n_components = H.shape[0]
    features, samples = V.shape
    # split train/test
    TRAIN_SIZE = 0.80
    if index is None:
      mask = np.random.rand(samples) < TRAIN_SIZE
    else:
      mask = index
    
    H_init = np.ones((n_components, samples))
    W_init = np.ones((features, n_components))/features

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
    lr=0.001,
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
    test_cret = nn.MSELoss(reduction="sum")
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
        # MSE Loss, ((x - y)**2).sum()
        test_out = deep_nmf(*test_input)
        test_loss.append((test_cret(test_out, data.h_test.tns)).item())
    return deep_nmf, loss_values, test_loss

def train_supervised_w(
    data: Alldata,
    num_layers,
    network_train_iterations,
    L1,
    L2,
    verbose=False,
    lr=0.001,
):
    v_train = data.v_train.tns
    # h_train = data.h_train.tns
    h_0_train = data.h_0_train.tns
    w_train = data.w.tns

    n_components = w_train.shape[0]
    features = v_train.shape[1]
    # build the architicture
    deep_nmf = SuperNet(num_layers, n_components, features, L1, L2)
    for w in deep_nmf.parameters():
        w.data.fill_(1.0)
    criterion = nn.MSELoss(reduction="mean")
    test_cret = nn.MSELoss(reduction="sum")
    optimizerADAM = optim.Adam(deep_nmf.parameters(), lr=lr)
    
    # Train the Network
    inputs = (h_0_train, v_train)
    test_input = (data.h_0_test.tns, data.v_test.tns)
    loss_values = []
    test_loss = []
    for i in range(network_train_iterations):
        out = deep_nmf(*inputs)
        loss = criterion(v_train, out.mm(w_train))  # loss between predicted and truth
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
        # MSE Loss, ((x - y)**2).sum()
        test_out = deep_nmf(*test_input)
        test_loss.append((test_cret(data.v_test.tns, test_out.mm(w_train))).item())
    return deep_nmf, loss_values, test_loss

def train_unsupervised(
    data: Alldata,
    num_layers,
    network_train_iterations,
    n_components,
    verbose=False,
    lr=0.0005,
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
    test = (data.h_0_test.tns, data.v_test.tns)
    dnmf_train_cost = []
    dnmf_test_cost = []
    for i in range(network_train_iterations):
        out = deep_nmf(*inputs)
        test_out = deep_nmf(*test)

        loss = cost_tns(v_train, dnmf_w, out, l_1, l_2)
        
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
        dnmf_test_cost.append(cost_tns(data.v_test.tns, dnmf_w, test_out,l_1, l_2).item())

    return deep_nmf, dnmf_train_cost, dnmf_test_cost, dnmf_w

def train_unsupervised_kl(
    data: Alldata,
    num_layers,
    network_train_iterations,
    n_components,
    verbose=False,
    lr=0.0005,
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
    test = (data.h_0_test.tns, data.v_test.tns)
    dnmf_train_cost = []
    dnmf_test_cost = []
    for i in range(network_train_iterations):
        out = deep_nmf(*inputs)
        test_out = deep_nmf(*test)

        loss = kl_reconstruct_error_tns(v_train, dnmf_w, out, l_1, l_2)
        
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
        dnmf_test_cost.append(kl_reconstruct_error_tns(data.v_test.tns, dnmf_w, test_out,l_1, l_2).item())

    return deep_nmf, dnmf_train_cost, dnmf_test_cost, dnmf_w

def train_supervised_known(
    data: Alldata,
    num_layers,
    network_train_iterations,
    L1,
    L2,
    verbose=False,
    lr=0.001,
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
    optimizerADAM = optim.Adam(deep_nmf.parameters(), lr=lr)
    
    # Train the Network
    inputs = (h_0_train, v_train)
    test_input = (data.h_0_test.tns, data.v_test.tns)
    loss_values = []
    test_loss = []
    for i in range(network_train_iterations):
        out = deep_nmf(*inputs)
        loss = cost_tns(v_train, data.w.tns, out)  

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
        # MSE Loss, ((x - y)**2).sum()
        test_out = deep_nmf(*test_input)
        test_loss.append(cost_tns(data.v_test.tns, data.w.tns, test_out).item())
    return deep_nmf, loss_values, test_loss


def plot_box(cols, labels, file_name,df1, title, xlabel, ylabel):
  plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
  m1 = np.log(df1[cols].values).mean(axis=0)
  fig = plt.figure()
  ax = fig.add_subplot(111)
  medianprops= dict(linestyle='', linewidth=2, color='blue')
  bp_dict = ax.boxplot(np.log(df1[cols].values),labels=labels,vert=True,showmeans=True,medianprops=medianprops)

  for i, line in enumerate(bp_dict['means']):
      # get position data for median line
      x, y = line.get_xydata()[0] # top of median line
      # overlay median value
      # text = ' μ={:.2f}'.format(m1[i])
      # ax.annotate(text, xy=(x, y))
      text(x, y, '%.1f' % m1[i]) # draw above, centered
  # ax.set_title(title)
  if xlabel is not None:
    ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  plt.savefig(f"plots/{file_name}.pdf")
  plt.show()


def train_reg_var(
    data: Alldata,
    num_layers,
    network_train_iterations,
    verbose=False,
    lr=0.001,
):
    v_train = data.v_train.tns
    h_train = data.h_train.tns
    h_0_train = data.h_0_train.tns

    n_components = h_train.shape[1]
    features = v_train.shape[1]
    # build the architicture
    deep_nmf = SuperNet_new(num_layers, n_components, features)
    for w in deep_nmf.parameters():
        w.data.fill_(1.0)
    criterion = nn.MSELoss(reduction="mean")
    test_cret = nn.MSELoss(reduction="sum")
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
        # MSE Loss, ((x - y)**2).sum()
        test_out = deep_nmf(*test_input)
        test_loss.append((test_cret(test_out, data.h_test.tns)/test_out.shape[0]).item())
    return deep_nmf, loss_values, test_loss

def train_unsupervised_opt1(
    data: Alldata,
    num_layers,
    network_train_iterations,
    n_components,
    verbose=False,
    lr=0.0005,
    l_1=0,
    l_2=0,
    include_reg = True
):
    v_train = data.v_train.tns
    h_0_train = data.h_0_train.tns
    features = v_train.shape[1]
    # build the architicture
    if include_reg:
        deep_nmf = UnsuperNetOpt1(num_layers, n_components, features, l_1, l_2)
    else:
        deep_nmf = UnsuperNetOpt1(num_layers, n_components, features, 0, 0)
    # initialize parameters
    dnmf_w = data.w_init.tns
    for w in deep_nmf.parameters():
        w.data.fill_(0.1)

    optimizerADAM = optim.Adam(deep_nmf.parameters(), lr=lr)
     # Train the Network
    inputs = (h_0_train, v_train)
    # test = (data.h_0_test.tns, data.v_test.tns, dnmf_w)
    dnmf_train_cost = []
    dnmf_test_cost = []
    for i in range(network_train_iterations):
        out = deep_nmf(*inputs)
        # test_out = deep_nmf(*test)

        loss = cost_tns(v_train, deep_nmf.W, out, l_1, l_2)
        
        if verbose:
            print(i, loss.item())

        optimizerADAM.zero_grad()
        loss.backward()
        optimizerADAM.step()

        # keep weights positive after gradient decent
        for w in deep_nmf.parameters():
            w.data = w.clamp(min=0,max=inf)
        # h_out = torch.transpose(out.data, 0, 1)
        # h_out_t = out.data

        
        dnmf_train_cost.append(loss.item())

        # test performance
        # dnmf_test_cost.append(cost_tns(data.v_test.tns, out[1], test_out[0],l_1, l_2).item())

    return deep_nmf, dnmf_train_cost, dnmf_test_cost, dnmf_w


def train_unsupervised_opt2(
    data: Alldata,
    num_layers,
    network_train_iterations,
    n_components,
    verbose=False,
    lr=0.0005,
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
    # test = (data.h_0_test.tns, data.v_test.tns, dnmf_w)
    dnmf_train_cost = []
    dnmf_test_cost = []
    for i in range(network_train_iterations):
        out = deep_nmf(*inputs)
        # test_out = deep_nmf(*test)

        loss = cost_tns(v_train, deep_nmf.W, out, l_1, l_2)
        
        if verbose:
            print(i, loss.item())

        optimizerADAM.zero_grad()
        loss.backward()
        optimizerADAM.step()

        # keep weights positive after gradient decent
        for w in deep_nmf.parameters():
            w.data = w.clamp(min=0,max=inf)
        # h_out = torch.transpose(out.data, 0, 1)
        # h_out_t = out.data

        
        dnmf_train_cost.append(loss.item())

        # test performance
        # dnmf_test_cost.append(cost_tns(data.v_test.tns, out[1], test_out[0],l_1, l_2).item())

    return deep_nmf, dnmf_train_cost, dnmf_test_cost, dnmf_w

def train_unsupervised_opt1_kl(
    data: Alldata,
    num_layers,
    network_train_iterations,
    n_components,
    verbose=False,
    lr=0.0005,
    l_1=0,
    l_2=0,
    include_reg = True
):
    v_train = data.v_train.tns
    h_0_train = data.h_0_train.tns
    features = v_train.shape[1]
    # build the architicture
    if include_reg:
        deep_nmf = UnsuperNetOpt1(num_layers, n_components, features, l_1, l_2)
    else:
        deep_nmf = UnsuperNetOpt1(num_layers, n_components, features, 0, 0)
    # initialize parameters
    for w in deep_nmf.parameters():
        w.data.fill_(0.1)

    optimizerADAM = optim.Adam(deep_nmf.parameters(), lr=lr)
     # Train the Network
    inputs = (h_0_train, v_train)
    test = (data.h_0_test.tns, data.v_test.tns)
    dnmf_train_cost = []
    dnmf_test_cost = []
    for i in range(network_train_iterations):
        out = deep_nmf(*inputs)
        loss = kl_reconstruct_error_tns(v_train, deep_nmf.W, out, l_1, l_2)
        
        if verbose:
            print(i, loss.item())

        optimizerADAM.zero_grad()
        loss.backward()
        optimizerADAM.step()

        # keep weights positive after gradient decent
        for w in deep_nmf.parameters():
            w.data = w.clamp(min=0,max=inf)
        
        dnmf_train_cost.append(loss.item())

        # test performance
        test_out = deep_nmf(*test)
        dnmf_test_cost.append(kl_reconstruct_error_tns(data.v_test.tns, deep_nmf.W, test_out,l_1, l_2).item())

    return deep_nmf, dnmf_train_cost, dnmf_test_cost, deep_nmf.W
