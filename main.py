import utils
import numpy as np
from sklearn.decomposition import NMF
import torch.optim as optim
import torch.utils.data as data_utils
from torch.utils.data.dataset import random_split
import torch
from my_layers import *
from matplotlib import pyplot as plt
from torchviz import make_dot

# Data Loading
M = np.load('synthetic_data/x.syn.many.types.0.5_sp.sp.npy')
X = M.T

# params
n_components = 21  # from summary table
samples, features = X.shape

# split train/test
TRAIN_SIZE = 0.80
mask = np.random.rand(samples) < TRAIN_SIZE

X_train = X[mask]
X_test = X[~mask]

# MU building target labels for training using Scikit NMF
nmf = NMF(n_components=n_components, solver='mu', beta_loss='frobenius', verbose=True)
W_train = nmf.fit_transform(X_train)
H = nmf.components_

# find exposures for the test set
W_test = nmf.transform(X_test)

# initialize exposures
W0_train = utils.initialize_transformed(X_train, n_components)
W0_test = utils.initialize_transformed(X_test, n_components)  # might be per sample or include the whole X ??

# Tensoring the Arrays
X_train_tensor = torch.from_numpy(X_train).float()
W_train_tensor = torch.from_numpy(W_train).float()
W0_train_tensor = torch.from_numpy(W0_train).float()

X_test_tensor = torch.from_numpy(X_test).float()
W_test_tensor = torch.from_numpy(W_test).float()
W0_test_tensor = torch.from_numpy(W0_test).float()

"""
Basic Model

5 layers with non-negative constrains on weights
Trained with projected Graident decent
"""

"""
                    Training The Network
"""

# get instances
constraints = utils.WeightClipper()
deep_nmf_model = MultiDNMFNet(15, n_components, features)
deep_nmf_model.apply(constraints)
criterion = nn.MSELoss()
optimizerSGD = optim.SGD(deep_nmf_model.parameters(), lr=1e-4)
optimizerADAM = optim.Adam(deep_nmf_model.parameters(), lr=1e-4)

inputs = (W0_train_tensor, X_train_tensor)
loss_values = []
for i in range(5000):
    out = deep_nmf_model(*inputs)
    loss = criterion(out, W_train_tensor)
    print(i, loss.item())

    optimizerADAM.zero_grad()
    loss.backward()
    optimizerADAM.step()

    deep_nmf_model.apply(constraints)  # keep wieghts positive
    loss_values.append(loss.item())

plt.plot(loss_values)

"""
Compare with Test Data

comparison is on the reconstruction Error
"""

test_inputs = (W0_test_tensor, X_test_tensor)
netwrok_prediction = deep_nmf_model(*test_inputs)

network_error = utils.frobinuis_reconstruct_error(X_test_tensor, netwrok_prediction, H)
print('deep NMF Error: ', network_error)

mu_error = utils.frobinuis_reconstruct_error(X_test_tensor, W_test_tensor, H)
print('regular MU Error: ', mu_error)

# make_dot(out, params=dict(deep_nmf_model.named_parameters())).render("attached", format="png")
