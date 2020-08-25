import time
import numpy as np
import math

inf = math.inf
EPSILON = np.finfo(np.float32).eps


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


def cost_function(v, w, h, l_1=0, l_2=0):
    d = (v - np.dot(w, h))
    return 0.5 * np.power(d, 2).sum() + l_1 * np.abs(h).sum() + 0.5 * l_2 * np.power(h, 2).sum()


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


def mu_update(V, W, H, l_1, l_2, update_H=True, update_W=True):
    # update W
    if update_W:
        W_nominator = np.dot(V, H.T)
        W_denominator = np.dot(W, np.dot(H, H.T)) + EPSILON
        delta = W_nominator / W_denominator
        W *= delta

    # update H
    if update_H:
        H_nominator = np.dot(W.T, V) - l_1
        H_denominator = np.dot(W.T.dot(W), H) + EPSILON + H * l_2
        delta = H_nominator / H_denominator
        H *= delta
    return W, H
