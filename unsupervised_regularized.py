import torch.optim as optim
from matplotlib import pyplot as plt
from my_layers import *
from utils import *
import pandas as pd
import sklearn.decomposition as sc
import nimfa as nm
from nimfa.methods.seeding import nndsvd
from scipy.optimize import nnls

EPSILON = np.finfo(np.float32).eps

# Data loading
signatures_df = pd.read_csv('data/simulated/ground.truth.syn.sigs.csv', sep=',')
exposures_df = pd.read_csv('data/simulated/ground.truth.syn.exposures.csv', sep=',')
category_df = pd.read_csv('data/simulated/ground.truth.syn.catalog.csv', sep=',')

# to use genetic synthitic data
W = signatures_df.iloc[:, 2:].values  # (f,k)
H = exposures_df.iloc[:, 1:].values  # (k,n)
V = category_df.iloc[:, 2:].values  # (f,n)

# to use simulated data
# W = abs(np.random.randn(96, 21))  # (f,k) normal
# H = abs(np.random.randn(21, 1350))  # (k,n) normal
# V = W.dot(H) + 0.1 * np.random.randn(96,1350)  # (f,n)

n_components = H.shape[0]
features, samples = V.shape

l_1 = 0
l_2 = 0

# split train/test
TRAIN_SIZE = 0.80
mask = np.random.rand(samples) < TRAIN_SIZE

# W_0, H_0 = nndsvd.Nndsvd().initialize(V[:, mask], n_components, {'flag': 0})

initialization = 'ones'
H_init = np.ones((n_components, samples))
W_init = np.ones((features, n_components))

# initialization = 'random'
# H_init = np.random.uniform(0, 0.1, (n_components, samples))
# W_init = np.random.uniform(0, 0.1, (features, n_components))
############################### Tensoring ###################################
v_train = torch.from_numpy(V[:, mask].T).float()
v_test = torch.from_numpy(V[:, ~mask].T).float()
h_0_train = torch.from_numpy(H_init[:, mask].T).float()
h_0_test = torch.from_numpy(H_init[:, ~mask].T).float()

if __name__ == "__main__":
    # setup params
    lr = 0.004  # for network GD
    num_layers = 12
    network_train_iteration = 400
    mu_iter = 400
    mu_test_iter = 50

    ############################ MU update both matrix ##########################
    h_mu = H_init[:, mask].copy()
    w_mu = W_init.copy()
    mu_training_cost = []
    for i in range(mu_iter):
        w_mu, h_mu = mu_update(V[:, mask], w_mu, h_mu, l_1, l_2)
        mu_training_cost.append(cost_function(V[:, mask], w_mu, h_mu, l_1, l_2))
    ############################# Deep NMF ###################################

    # build the architicture
    constraints = WeightClipper(lower=0)
    deep_nmf = RegNet(num_layers, n_components, features, l_1, l_2)
    dnmf_w = torch.from_numpy(W_init.T).float()
    deep_nmf.apply(constraints)
    criterion = nn.MSELoss()

    optimizerADAM = optim.Adam(deep_nmf.parameters(), lr=lr)

    # Train the Network
    inputs = (h_0_train, v_train)
    dnmf_train_cost = []
    for i in range(network_train_iteration):
        out = deep_nmf(*inputs)

        R = v_train - out.mm(dnmf_w)
        loss = 0.5 * torch.sum(torch.mul(R, R)) + l_1 * out.sum() + 0.5 * l_2 * out.pow(2).sum()

        print(i, loss.item())

        optimizerADAM.zero_grad()
        loss.backward()
        optimizerADAM.step()

        deep_nmf.apply(constraints)  # keep weights positive after gradient decent
        h_out = torch.transpose(out.data, 0, 1)
        h_out_t = out.data

        # NNLS
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html

        w_arrays = [nnls(out.data.numpy(), V[f, mask])[0] for f in range(features)]
        nnls_w = np.stack(w_arrays, axis=-1)
        dnmf_w = torch.from_numpy(nnls_w).float()

        # dnmf_w = dnmf_w * (h_out.mm(v_train)).div(h_out.mm(h_out_t).mm(dnmf_w))
        dnmf_train_cost.append(loss.item())
    ################################### Testing ###################################
    ################################### MU update new H ###########################

    h_mu_test = H_init[:, ~mask].copy()
    mu_test_cost = []
    for i in range(mu_test_iter):
        _, h_mu_test = mu_update(V[:, ~mask], w_mu, h_mu_test, l_1, l_2, update_W=False)
        mu_test_cost.append(cost_function(V[:, ~mask], w_mu, h_mu_test, l_1, l_2))

    ################################## DNMF Prediction ############################
    test_inputs = (h_0_test, v_test)
    start_iter = time.time()
    netwrok_prediction = deep_nmf(*test_inputs)
    dnmf_elapsed = round(time.time() - start_iter, 5)
    D = v_test - netwrok_prediction.mm(dnmf_w)
    dnmf_test_cost = (0.5 * torch.sum(
        torch.mul(D, D)) + l_1 * netwrok_prediction.sum() + 0.5 * l_2 * netwrok_prediction.pow(2).sum()).item()

    epochs = range(0, network_train_iteration - 1)
    plt.semilogy(mu_training_cost, '-*', label='Training loss mu')
    plt.semilogy(dnmf_train_cost, '-*', label='Training loss DNN')
    plt.title(f"Beta=2, DNMF Vs MU")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.annotate(
        f'PARAMS: \n L1={l_1} \n L2={l_2} \n layers={num_layers} \n MU iter={mu_test_iter} \n Initialization={initialization} \n '
        f'Results: \n ' f'DN_COST={dnmf_test_cost} \n MU_COST={round(mu_test_cost[-1])}',
        xy=(0.6, 0.4), xycoords='axes fraction')
    plt.show()
