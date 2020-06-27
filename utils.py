import time

import numpy as np
import math

inf = math.inf


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


def initialize_exposures(V, n_components, method='random', seed=1984):
    """
    Average :
    this is initialization of w matrix as in scikiit transform method
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html#sklearn.decomposition.NMF.transform
    """
    n_features, n_samples = V.shape
    avg = np.sqrt(V.mean() / n_components)
    rng = np.random.RandomState(seed)

    if method == 'random':
        exposures = avg * rng.rand(n_components, n_samples)
    elif method == 'ones':
        exposures = np.ones((n_components, n_samples))
    elif method == 'average':
        exposures = np.full((n_components, n_samples), avg)
    return exposures


def frobinuis_reconstruct_error(x, w, h):
    return np.linalg.norm(x - np.dot(w, h))


def kl_reconstruct_error(x, w, h):
    a = x
    b = w.dot(h)
    return np.sum(
        np.where(a != 0, np.where(b != 0, a * np.log(a / b) - a + b, 0), 0)
    )  # from wp, zeros aren't counted
    # np.sqrt(2*np.sum(np.where(a != 0, np.where(b != 0, a * np.log(a / b) - a + b, 0), 0))) at Scikit, incorrect imo


def sBCD_update(V, W, H, O, obj="kl"):
    """
    Fast beta-divergence update, using: Fast Bregman Divergence NMF using Taylor Expansion and Coordinate Descent, Li 2012
    O parameter is a masking matrix, for cross-validation purposes, pass O=1
    """
    n, m = V.shape
    K = W.shape[1]
    V_tag = np.dot(W, H)
    E = np.subtract(V, V_tag)

    if obj == "kl":
        B = np.divide(1, V_tag) * O
    elif obj == "euc":
        B = np.ones((V.shape)) * O
    else:  # obj == 'is' Itakura-Saito
        B = np.divide(1, V_tag ** 2) * O

    for k in range(K):
        V_k = np.add(E, np.dot(W[:, k].reshape((n, 1)), H[k, :].reshape((1, m))))
        B_V_k = B * V_k
        # update H
        H[k] = np.maximum(
            1e-16, (np.dot(B_V_k.T, W[:, k])) / (np.dot(B.T, W[:, k] ** 2))
        )
        # update W
        W[:, k] = np.maximum(
            1e-16, (np.dot(B_V_k, H[k])) / (W[:, k] + np.dot(B, H[k] ** 2))
        )
        E = np.subtract(V_k, np.dot(W[:, k].reshape((n, 1)), H[k, :].reshape((1, m))))

    return W, H


class NMF:
    def __init__(self, rank=10, **kwargs):

        self._rank = rank

    def initialize_w(self):
        """ Initalize W to random values [0,1]."""

        self.W = np.random.random((self.X_dim, self._rank))

    def initialize_h(self):
        """ Initalize H to random values [0,1]."""

        self.H = np.random.random((self._rank, self._samples))
        self.H_init = self.H.copy()

    def check_non_negativity(self):

        if self.X.min() < 0:
            return 0
        else:
            return 1

    def update_h(self):

        XtW = np.dot(self.W.T, self.X)
        HWtW = np.dot(self.W.T.dot(self.W), self.H) + 2 ** -8
        self.H *= XtW
        self.H /= HWtW

    def update_w(self):

        XH = self.X.dot(self.H.T)
        WHtH = self.W.dot(self.H.dot(self.H.T)) + 2 ** -8
        self.W *= XH
        self.W /= WHtH

    def compute_factors(self, X, max_iter=100, update_W=True, update_H=True):
        self.X = X
        self.X_dim, self._samples = self.X.shape

        if self.check_non_negativity():
            pass
        else:
            print("The given matrix contains negative values")
            exit()

        if not hasattr(self, "W") and update_W:
            self.initialize_w()

        if not hasattr(self, "H") or update_H:
            self.initialize_h()

        self.frob_error = np.zeros(max_iter)
        start_iter = time.time()
        for i in range(max_iter):
            if update_W:
                self.update_w()

            self.update_h()

            self.frob_error[i] = frobinuis_reconstruct_error(self.X, self.W, self.H)
        self.elapsed = time.time() - start_iter


if __name__ == "__main__":
    # X = np.random.random((10, 10))
    X = np.load("data/synthetic_data/x.syn.many.types.0.5_sp.sp.npy")
    nmf = NMF(rank=21)
    nmf.compute_factors(X, 200)
    print(nmf.W.shape, nmf.H.shape)
    print(nmf.frob_error)
    print(nmf.elapsed)
