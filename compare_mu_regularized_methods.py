from utils import *
import pandas as pd
import sklearn.decomposition as sc
from matplotlib import pyplot as plt

EPSILON = np.finfo(np.float32).eps

# Data loading
signatures_df = pd.read_csv("data/simulated/ground.truth.syn.sigs.csv", sep=",")
exposures_df = pd.read_csv("data/simulated/ground.truth.syn.exposures.csv", sep=",")
category_df = pd.read_csv("data/simulated/ground.truth.syn.catalog.csv", sep=",")

# to use genetic synthytic data
W = signatures_df.iloc[:, 2:].values  # (f,k)
H = exposures_df.iloc[:, 1:].values  # (k,n)
V = category_df.iloc[:, 2:].values  # (f,n)

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

print('Hoyer method: ', error_hoyer[-1])

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

print('our method: ', our_error[-1])

plt.semilogy(error_hoyer, '-*', label='Hoyer')
plt.semilogy(our_error, '-*', label='Our')
plt.legend()
plt.show()
