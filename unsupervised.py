import torch.optim as optim
from matplotlib import pyplot as plt
from my_layers import *
from utils import *
import pandas as pd

EPSILON = np.finfo(np.float32).eps

# Data loading
signatures_df = pd.read_csv('data/simulated/ground.truth.syn.sigs.csv', sep=',')
exposures_df = pd.read_csv('data/simulated/ground.truth.syn.exposures.csv', sep=',')
category_df = pd.read_csv('data/simulated/ground.truth.syn.catalog.csv', sep=',')

# W = signatures_df.iloc[:, 2:].values  # (f,k)
# H = exposures_df.iloc[:, 1:].values  # (k,n)
# V = category_df.iloc[:, 2:].values  # (f,n)
W = abs(np.random.randn(96, 21))  # (f,k) normal
H = abs(np.random.randn(21, 1350))  # (k,n) normal
V = W.dot(H)  # (f,n)

n_components = H.shape[0]
features, samples = V.shape

H_init = initialize_exposures(V, n_components, method='ones')  # (n,k)
# split train/test
TRAIN_SIZE = 0.80
mask = np.random.rand(samples) < TRAIN_SIZE

############################### Tensoring ###################################
v_train = torch.from_numpy(V[:, mask].T).float()
v_test = torch.from_numpy(V[:, ~mask].T).float()
h_train = torch.from_numpy(H[:, mask].T).float()
h_0_train = torch.from_numpy(H_init[:, mask].T).float()
h_0_test = torch.from_numpy(H_init[:, ~mask].T).float()
W_tensor = torch.from_numpy(W.T).float()

if __name__ == "__main__":
    # setup params
    lr = 0.0001
    num_layers = 5
    network_train_iteration = 1000
    mu_iter = 50

    ############################ MU update exposures ##########################
    w = W.copy()
    h = H_init[:, ~mask].copy()
    v = V[:, ~mask].copy()
    mu_error = np.zeros(mu_iter)

    start_iter = time.time()
    for i in range(mu_iter):
        nominator = np.dot(w.T, v)
        denominator = np.dot(w.T.dot(w), h) + EPSILON
        h *= nominator
        h /= denominator
        mu_error[i] = round(frobinuis_reconstruct_error(v, w, h), 0)
    mu_elapsed = round(time.time() - start_iter, 5)

    ############################# Deep NMF ###################################

    # build the architicture
    constraints = WeightClipper(lower=0)
    deep_nmf = UnSuperNet(num_layers, n_components, features)
    deep_nmf.apply(constraints)
    criterion = nn.MSELoss(reduction='mean')

    optimizerADAM = optim.Adam(deep_nmf.parameters(), lr=lr)

    # Train the Network
    inputs = (h_0_train, v_train)
    loss_values = []
    for i in range(network_train_iteration):
        out = deep_nmf(*inputs)
        loss = criterion(out, h_train)  # loss between predicted and truth
        # loss = criterion(out.mm(W_tensor), v_train) # reconstruction loss
        print(i, loss.item())

        optimizerADAM.zero_grad()
        loss.backward()
        optimizerADAM.step()

        deep_nmf.apply(constraints)  # keep wieghts positive after gradient decent
        loss_values.append(loss.item())

    test_inputs = (h_0_test, v_test)
    start_iter = time.time()
    netwrok_prediction = deep_nmf(*test_inputs)
    dnmf_elapsed = round(time.time() - start_iter, 5)
    dnmf_err = round(frobinuis_reconstruct_error(v, w, netwrok_prediction.detach().numpy().T), 2)

    epochs = range(0, network_train_iteration - 1)
    # plt.semilogy(mu_training_loss, '-*', label='Training loss mu')
    plt.semilogy(loss_values, '-*', label='Training loss DNN')
    plt.title(f"Beta=2, weights not shared")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.annotate(
        f'PARAMS: \n lr={lr} \n layers={num_layers} \n Train_iter={network_train_iteration} \n Results: \n DNMF_Error={dnmf_err} \n MU_Error={mu_error[-1]} \n DNMF_time={dnmf_elapsed} \n MU_time={mu_elapsed}',
        xy=(0.68, 0.5), xycoords='axes fraction')
    plt.show()
