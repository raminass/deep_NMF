import numpy as np


class WeightClipper(object):
    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, "weight"):
            w = module.weight.data
            w = w.clamp(min=0)
            module.weight.data = w


def initialize_transformed(x, n_components):
    """
    this is initialization of w matrix as in scikiit transform method
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html#sklearn.decomposition.NMF.transform
    """
    avg = np.sqrt(x.mean() / n_components)
    w = np.full((x.shape[0], n_components), avg)
    return w


def frobinuis_reconstruct_error(x, w, h):
    x = x.data.numpy()
    w = w.data.numpy()
    reconstructed = x - w.dot(h)
    return np.linalg.norm(reconstructed)


def KL(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)
    return np.sum(
        np.where(a != 0, np.where(b != 0, a * np.log(a / b) - a + b, 0), 0)
    )  # from wp, zeros aren't counted


# Fast beta-divergence update, using: Fast Bregman Divergence NMF using Taylor Expansion and Coordinate Descent, Li 2012
# O parameter is a masking matrix, for cross-validation purposes, pass O=1
def sBCD_update(V, W, H, O, obj="kl"):
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
