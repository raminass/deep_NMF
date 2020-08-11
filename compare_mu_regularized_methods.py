from utils import *
import pandas as pd
import sklearn.decomposition as sc

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

for i in range(200):
    nominator = np.dot(W.T, V)
    denominator = np.dot(W.T.dot(W), H) + EPSILON*C + C * lam
    delta = nominator / denominator
    H *= delta

print('Hoyer method: ', frobenius_reconstruct_error(V, W, H))

# Our method
H = H_init.copy()
C = initialize_exposures(V, n_components, method="ones")
lam = 1.5

for i in range(200):
    nominator = np.dot(W.T, V) - C * lam
    denominator = np.dot(W.T.dot(W), H) + EPSILON*C
    delta = nominator / denominator
    H *= delta

print('our method: ', frobenius_reconstruct_error(V, W, H))
