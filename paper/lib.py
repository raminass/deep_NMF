from matplotlib import pyplot as plt
from matplotlib.pyplot import text
# from my_layers import UnsuperNet, SuperNet, UnsuperNetOpt3
import utils as util
import pandas as pd
import sklearn.decomposition as sc
import numpy as np
import matplotlib.ticker as mticker
import joblib
import seaborn as sns
from sklearn.model_selection import KFold

plt.rcParams['figure.figsize'] = (15.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# # use LaTeX fonts in the plot
# plt.rc('text', usetex=True)
# plt.rc('font', size=30)
# plt.rc('legend', fontsize=20)