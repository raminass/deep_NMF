from utils import *
import pandas as pd
import sklearn.decomposition as sc
from matplotlib import pyplot as plt

EPSILON = np.finfo(np.float32).eps

hoyer_sim = []
our_sim = []
for j in range(10):
    # to use simulated data
    W = abs(np.random.randn(96, 21))  # (f,k) normal
    H = abs(np.random.randn(21, 1350))  # (k,n) normal
    V = W.dot(H) + 0.1 * np.random.randn(96, 1350)  # (f,n)

    n_components = H.shape[0]
    features, samples = V.shape

    H_init = initialize_exposures(V, n_components, method="ones")  # (n,k)

    # Hoyer Method L1

    H = H_init.copy()
    C = initialize_exposures(V, n_components, method="ones")
    lam = 1.5
    error_hoyer = []
    for i in range(200):
        nominator = np.dot(W.T, V)
        denominator = np.dot(W.T.dot(W), H) + EPSILON * C + C * lam
        delta = nominator / denominator
        H *= delta
        error_hoyer.append(frobenius_reconstruct_error(V, W, H) + np.linalg.norm((H), ord=1))

    # Our method
    H = H_init.copy()
    C = initialize_exposures(V, n_components, method="ones")
    lam = 1.5
    our_error = []

    for i in range(200):
        nominator = np.dot(W.T, V) - C * lam
        denominator = np.dot(W.T.dot(W), H) + EPSILON * C
        delta = nominator / denominator
        H *= delta
        our_error.append(frobenius_reconstruct_error(V, W, H) + np.linalg.norm((H), ord=1))

    hoyer_sim.append(error_hoyer[-1])
    our_sim.append(our_error[-1])

print('Hoyer method: ', hoyer_sim)
print('our method: ', our_sim)

plt.semilogy(hoyer_sim, '-*', label='Hoyer')
plt.semilogy(our_sim, '-*', label='Our')
plt.legend()
plt.show()
