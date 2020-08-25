from utils import *
import pandas as pd
import sklearn.decomposition as sc
from matplotlib import pyplot as plt

EPSILON = np.finfo(np.float32).eps
lam = 5

hoyer_sim = []
our_sim = []
normal_sim = []
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
    error_hoyer = []
    for i in range(200):
        nominator = np.dot(W.T, V)
        denominator = np.dot(W.T.dot(W), H) + EPSILON + lam
        delta = nominator / denominator
        H *= delta
        error_hoyer.append(cost_function(V, W, H, l_1=lam, l_2=0))

    # Our method
    H = H_init.copy()
    our_error = []

    for i in range(200):
        nominator = np.dot(W.T, V) - lam
        denominator = np.dot(W.T.dot(W), H) + EPSILON
        delta = nominator / denominator
        H *= delta
        our_error.append(cost_function(V, W, H, l_1=lam, l_2=0))

    # normal update - no regularizer
    H = H_init.copy()
    normal_error = []
    for i in range(200):
        nominator = np.dot(W.T, V)
        denominator = np.dot(W.T.dot(W), H) + EPSILON
        delta = nominator / denominator
        H *= delta
        normal_error.append(cost_function(V, W, H, l_1=lam, l_2=0))

    hoyer_sim.append(error_hoyer[-1])
    our_sim.append(our_error[-1])
    normal_sim.append(normal_error[-1])

print('Hoyer method: ', hoyer_sim)
print('our method: ', our_sim)
print('Normal method: ', normal_sim)

plt.semilogy(hoyer_sim, '-*', label='Hoyer')
plt.semilogy(our_sim, '-*', label='Our')
plt.semilogy(normal_sim, '-*', label='Normal')
plt.legend()
plt.show()
