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

H_init = initialize_exposures(V, n_components, method='average')  # (n,k)

# split train/test
TRAIN_SIZE = 0.80
mask = np.random.rand(samples) < TRAIN_SIZE

W_0, H_0 = nndsvd.Nndsvd().initialize(V[:, mask], n_components, {'flag': 0})
############################### Tensoring ###################################
v_train = torch.from_numpy(V[:, mask].T).float()
v_test = torch.from_numpy(V[:, ~mask].T).float()
h_train = torch.from_numpy(H[:, mask].T).float()
h_0_train = torch.from_numpy(H_init[:, mask].T).float()
# h_0_train = torch.from_numpy(H_0.T).float()
h_0_test = torch.from_numpy(H_init[:, ~mask].T).float()

if __name__ == "__main__":
    # setup params
    lr = 0.002
    num_layers = 10
    network_train_iteration = 1000
    mu_iter = 50

    ############################ MU update exposures ##########################
    model = nm.Nmf(V[:, mask], rank=n_components, max_iter=1000, track_error=True)
    mu_fit = model.factorize()
    mu_training_loss = np.sqrt(model.tracker.get_error())

    ############################# Deep NMF ###################################

    # build the architicture
    constraints = WeightClipper(lower=0)
    deep_nmf = MultiFrDNMFNet(num_layers, n_components, features)
    dnmf_w = torch.nn.init.uniform_(torch.Tensor(n_components, features), 0, 1)
    # dnmf_w = torch.((n_components, features))
    # dnmf_w = torch.from_numpy(W_0.T).float()
    deep_nmf.apply(constraints)
    criterion = nn.MSELoss()

    optimizerADAM = optim.Adam(deep_nmf.parameters(), lr=lr)

    # Train the Network
    inputs = (h_0_train, v_train)
    loss_values = []
    for i in range(network_train_iteration):
        out = deep_nmf(*inputs)

        R = v_train - out.mm(dnmf_w)
        loss = torch.sum(torch.mul(R, R))

        # loss = criterion(v_train, out.mm(dnmf_w))  # loss between predicted and truth

        print(i, loss.item())

        optimizerADAM.zero_grad()
        loss.backward()
        optimizerADAM.step()

        deep_nmf.apply(constraints)  # keep wieghts positive after gradient decent
        h_out = torch.transpose(out.data, 0, 1)
        h_out_t = out.data

        # NNLS 
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html
        
        w_arrays = [nnls(out.data.numpy(),V[f, mask])[0] for f in range(features)]
        nnls_w = np.stack(w_arrays, axis=-1)
        dnmf_w = torch.from_numpy(nnls_w).float()
        
        # dnmf_w = dnmf_w * (h_out.mm(v_train)).div(h_out.mm(h_out_t).mm(dnmf_w))
        loss_values.append(loss.item())



    # test_inputs = (h_0_test, v_test)
    # start_iter = time.time()
    # netwrok_prediction = deep_nmf(*test_inputs)
    # dnmf_elapsed = round(time.time() - start_iter, 5)
    # dnmf_err = round(
    #     frobinuis_reconstruct_error(V[:, ~mask], dnmf_w.data.numpy().T, netwrok_prediction.data.numpy().T), 2)
    # mu_error = round(frobinuis_reconstruct_error(V[:, ~mask], W, fit.coef()), 2)

    frobenius_reconstruct_error(V[:, mask], dnmf_w.data.numpy().T, out.data.numpy().T)
    frobenius_reconstruct_error(V[:, mask], mu_fit.basis(), mu_fit.coef())

    epochs = range(0, network_train_iteration - 1)
    plt.semilogy(mu_training_loss, '-*', label='Training loss mu')
    plt.semilogy(np.sqrt(loss_values), '-*', label='Training loss DNN')
    plt.title(f"Beta=2, DNMF Vs MU")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # plt.annotate(
    #     f'PARAMS: \n W_shared={False} \n lr={lr} \n layers={num_layers} \n Train_iter={network_train_iteration} \n Results: \n DNMF_Error={dnmf_err} \n MU_Error={mu_error} \n DNMF_time={dnmf_elapsed} \n MU_time={mu_elapsed}',
    #     xy=(0.68, 0.5), xycoords='axes fraction')
    plt.show()
