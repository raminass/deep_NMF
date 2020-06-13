# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# This is Multi layer net using the general beta MU

# %%
import utils
import numpy as np
import sklearn.decomposition as sk_deco
import torch.optim as optim
import torch.utils.data as data_utils
from torch.utils.data.dataset import random_split
import torch
from my_layers import *
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
import torch.nn as nn

# %%
# Data Loading
M = np.load("synthetic_data/x.syn.many.types.0.5_sp.sp.npy")
X = M.T


# %%
# params
n_components = 21  # from summary table
samples, features = X.shape


# %%
# split train/test
TRAIN_SIZE = 0.80
mask = np.random.rand(samples) < TRAIN_SIZE

X_train = X[mask]
X_test = X[~mask]


# %%
# MU building target labels for training using Scikit NMF
nmf = sk_deco.NMF(
    n_components=n_components, solver="mu", beta_loss="kullback-leibler", verbose=True
)
W_train = nmf.fit_transform(X_train)
H = nmf.components_


# %%
W_test = nmf.transform(X_test)

# %% [markdown]
# #### initialize exposures
#
#

# %%
W0_train = utils.initialize_transformed(X_train, n_components)
W0_test = utils.initialize_transformed(
    X_test, n_components
)  # might be per sample or include the whole X ??

# %% [markdown]
# #### Tensoring the Arrays

# %%
X_train_tensor = torch.from_numpy(X_train).float()
W_train_tensor = torch.from_numpy(W_train).float()
W0_train_tensor = torch.from_numpy(W0_train).float()

X_test_tensor = torch.from_numpy(X_test).float()
W_test_tensor = torch.from_numpy(W_test).float()
W0_test_tensor = torch.from_numpy(W0_test).float()

# %% [markdown]
# ## Basic Model
# 9 layers with non-negative constrains on weights
#
# Trained with Graident decent
# %% [markdown]
# ### Training The Network

# %%
constraints = utils.WeightClipper()
beta_nmf = MultiBetaDNMFNet(12, 1, n_components, features)
beta_nmf.apply(constraints)
criterion = nn.MSELoss()
optimizerSGD = optim.SGD(beta_nmf.parameters(), lr=1e-4)
optimizerADAM = optim.Adam(beta_nmf.parameters(), lr=1e-4)

# %%
inputs = (W0_train_tensor, X_train_tensor)
loss_values = []
for i in range(3000):

    out = beta_nmf(*inputs)
    loss = criterion(out, W_train_tensor)
    print(i, loss.item())

    optimizerADAM.zero_grad()
    loss.backward()
    optimizerADAM.step()

    beta_nmf.apply(constraints)  # keep wieghts positive
    loss_values.append(loss.item())


# %%
plt.plot(loss_values)

# %% [markdown]
# ### Compare with Test Data
# comparison is on the reconstruction Error

# %%
test_inputs = (W0_test_tensor, X_test_tensor)
netwrok_prediction = beta_nmf(*test_inputs)


# %%
network_error = utils.frobinuis_reconstruct_error(X_test_tensor, netwrok_prediction, H)
error = sk_deco._nmf._beta_divergence(
    X_test_tensor.data.numpy(), netwrok_prediction.data.numpy(), H, 1
)
print("deep NMF Error: ", network_error)


# %%
mu_error = utils.frobinuis_reconstruct_error(X_test_tensor, W_test_tensor, H)
print("regular MU Error: ", mu_error)

# change the frobinious reconstruction error to KL

# %%
model_1 = BetaNMFLayer(1, n_components, features)
x = (W0_train_tensor[1], X_train_tensor[1])
with torch.onnx.set_training(model_1, False):
    trace, _ = torch.jit._get_trace_graph(model_1, args=(W0_train_tensor[1], X_train_tensor[1],))
make_dot_from_trace(trace)
torch.ji
# make_dot(out, params=dict(deep_nmf_model.named_parameters()))
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('runs/beta_nmf')
# writer.add_graph(net, ima


