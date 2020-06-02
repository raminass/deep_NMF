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
