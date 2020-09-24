# %% [markdown]
# Refs
"""
networks: https://nx.github.io/documentation/stable/tutorial.html
sparse matrix: https://machinelearningmastery.com/sparse-matrices-for-machine-learning/
mygene: https://nbviewer.jupyter.org/gist/newgene/6771106
        https://docs.mygene.info/projects/mygene-py/en/latest/
article: chrome-extension://bomfdkbfpdhijjbeoicnfhjbdhncfhig/view.html?mp=7cLxK10j
        https://www.nature.com/articles/s41586-020-2286-9#Sec36
"""

# %%
import numpy as np
import pandas as pd
import scipy as sp
import math
import networkx as nx
from sklearn.linear_model import LinearRegression
import sklearn as sc
import plotly.graph_objects as go
import mygene
import random
import joblib
import statsmodels.api as sm
import plotly.express as px
from network_random import *
import os
from collections import defaultdict

%load_ext autoreload
%autoreload 2

# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"

mg = mygene.MyGeneInfo()  # api to map genes code

# %%
# Global Variables
PROPAGATE_ALPHA = 0.9
PROPAGATE_ITERATIONS = 200
PROPAGATE_EPSILON = 10 ** (-4)
target = [
    1003,
    1000,
    1001,
    1002,
    1500,
    1499,
    1495,
    100506658,
    7122,
    7082,
    9414,
    27134,
]  # vascular protiens

target_names = [
    "Cadherin-5",
    "Cadherin-2",
    "Cadherin-3",
    "Cadherin-4",
    "Catenin delta 1",
    "Cateninβ  ",
    "α Catenin ",
    "Occludin",
    "Claudin-5",
    "ZO-1",
    "ZO-2",
    "ZO-3",
]

# %%


def read_network(network_filename):
    network = pd.read_table(network_filename, header=None, usecols=[0, 1, 2])
    return nx.from_pandas_edgelist(network, 0, 1, 2)


def generate_similarity_matrix(network):
    genes = sorted(network.nodes)
    matrix = nx.to_scipy_sparse_matrix(network, genes, weight=2)

    norm_matrix = sp.sparse.diags(1 / sp.sqrt(matrix.sum(0).A1), format="csr")
    matrix = norm_matrix * matrix * norm_matrix

    return PROPAGATE_ALPHA * matrix, genes


def propagate(seeds, matrix, gene_indexes, num_genes):
    F_t = np.zeros(num_genes)
    F_t[[gene_indexes[seed] for seed in seeds if seed in gene_indexes]] = 1
    Y = (1 - PROPAGATE_ALPHA) * F_t

    for _ in range(PROPAGATE_ITERATIONS):
        F_t_1 = F_t
        F_t = matrix.dot(F_t_1) + Y

        if math.sqrt(sp.linalg.norm(F_t_1 - F_t)) < PROPAGATE_EPSILON:
            break

    return F_t


def generate_propagate_data(network, interactors=None):
    matrix, genes = generate_similarity_matrix(network)
    num_genes = len(genes)
    gene_indexes = dict([(gene, index) for (index, gene) in enumerate(genes)])
    if interactors:
        gene_scores = {gene: propagate(
            [gene], matrix, gene_indexes, num_genes) for gene in interactors}
    else:
        gene_scores = {gene: propagate(
            [gene], matrix, gene_indexes, num_genes) for gene in genes}

    return matrix, num_genes, gene_indexes, gene_scores


# %%
# Loading Datasets
g = read_network("H_sapiens.net")
# https://www.nature.com/articles/s41586-020-2286-9#Sec36
interactions = pd.read_csv("interactions.csv")
perm3 = pd.read_csv("perm3.csv")  # permeability after 3 Days
perm4 = pd.read_csv("perm4.csv")  # permeability after 4 Days

# %%
# propogate Network for all Genes
network_path = f"network_scores.pkl.gz"

if os.path.exists(network_path):
    print('loading propagated network from disk')
    network_scores = joblib.load(network_path)
    W, num_genes, gene_indexes, gene_scores = (
        network_scores["W"],
        network_scores["num_genes"],
        network_scores["gene_indexes"],
        network_scores["gene_scores"],
    )
else:
    print('start propagating network')
    W, num_genes, gene_indexes, gene_scores = generate_propagate_data(g)
    network_scores = {"W": W, "num_genes": num_genes, "gene_indexes": gene_indexes, "gene_scores": gene_scores}
    joblib.dump(network_scores, network_path)

target_index = [gene_indexes[i] for i in target if i in gene_indexes]

# Network stats
g_nodes, g_degrees = zip(*g.degree())
fig = px.histogram(x=g_degrees, nbins=1000)
fig.show()

# %%
# mapping to entrez id
xli = interactions["PreyGene"].unique().tolist()
out = pd.DataFrame(mg.querymany(xli, scopes="symbol", fields="entrezgene", species="human"))
interactions = pd.merge(interactions, out[["query", "entrezgene"]], left_on="PreyGene", right_on="query")
interactions["entrezgene"] = interactions["entrezgene"].astype(int)

# %%
#  random networks for interaction genes 332
E = g.number_of_edges()
Q = 10

inter_genes = list(interactions["entrezgene"].unique())
random_networks = {}
for i in range(100):
    H = g.copy()
    # rand_g, swaps = randomize_by_edge_swaps(g, 10)
    nx.swap.double_edge_swap(H, nswap=Q*E, max_tries=Q*E*2)
    W_temp, num_genes_temp, gene_indexes_temp, gene_scores_temp = generate_propagate_data(H, inter_genes)
    random_networks[i] = gene_scores_temp
    print(f"network {i} generated")

random_networks_path = f"random_networks_score.pkl.gz"
joblib.dump(random_networks, random_networks_path)

# %%
# grouping count
interaction_count = interactions.groupby("viral", as_index=False).count()
interaction_count["viral"] = interaction_count.apply(lambda row: row["viral"].lower(), axis=1)

# %%
# lower case the protiens names
perm3["Viral Protein SARS-Cov-2"] = perm3.apply(lambda row: row["Viral Protein SARS-Cov-2"].lower(), axis=1)
perm4["Viral Protein SARS-Cov-2"] = perm4.apply(lambda row: row["Viral Protein SARS-Cov-2"].lower(), axis=1)

# join permeabilty and interactions
corr_data = interaction_count.merge(perm3, left_on="viral", right_on="Viral Protein SARS-Cov-2")
corr_data = corr_data.merge(perm4, left_on="viral", right_on="Viral Protein SARS-Cov-2")

ongoing_data = pd.DataFrame(columns=['viral', 'interactions_count', 'perm_3_days'])
ongoing_data['viral'] = corr_data['viral']
ongoing_data['interactions_count'] = corr_data['Preys']
ongoing_data['perm_3_days'] = corr_data['Permeability Value (3 days)']
ongoing_data['perm_4_days'] = corr_data['Permeability Value (4 days)']
del(corr_data)

# %%
# calculate p-value and scores by random sources
viral_scores = []
viral_p_values = []
for index, row in ongoing_data.iterrows():
    runs = []
    # calculate the score of viral gene
    sources = interactions[interactions["viral"].str.lower() == row["viral"]]["entrezgene"].to_list()
    score = sum([sum(v[target_index])for k, v in gene_scores.items() if k in sources]) / len(sources)
    runs.append(score)
    viral_scores.append(score)

    # generate 100 random scores
    for i in range(100):
        random_sources = random.sample(population=list(g.nodes), k=len(sources))
        score = sum([sum(v[target_index]) for k, v in gene_scores.items() if k in random_sources]) / len(sources)
        runs.append(score)

    pvalue = (102 - sp.stats.rankdata(runs, method="ordinal")[0]) / 101
    viral_p_values.append(pvalue)

ongoing_data["network_score"] = viral_scores
ongoing_data["pvalue"] = viral_p_values
ongoing_data["adjusted_pvalue"] = sm.stats.multipletests(ongoing_data["pvalue"], alpha=0.05, method="fdr_bh", is_sorted=False)[1]

# %%
# calculate p-value and scores multi-Nets
viral_p_values = []
for index, row in ongoing_data.iterrows():
    runs = []
    # calculate the score of viral gene
    sources = interactions[interactions["viral"].str.lower() == row["viral"]]["entrezgene"].to_list()
    score = sum([sum(v[target_index]) for k, v in gene_scores.items() if k in sources]) / len(sources)
    runs.append(score)

    # generate 100 random scores by 100 random networks
    for net, prop in random_networks.items():
        score = sum([sum(v[target_index]) for k, v in prop.items() if k in sources]) / len(sources)
        runs.append(score)

    pvalue = (102 - sp.stats.rankdata(runs, method="ordinal")[0]) / 101
    viral_p_values.append(pvalue)

ongoing_data["pvalue_multi_net"] = viral_p_values
ongoing_data["adjusted_pvalue_multi_net"] = sm.stats.multipletests(ongoing_data["pvalue_multi_net"], alpha=0.05, method="fdr_bh", is_sorted=False)[1]

# %%
# random neighbours with same degree
genes_by_degree = defaultdict(list)
degrees_dict = dict(g.degree)
for key, val in sorted(degrees_dict.items()):
    genes_by_degree[val].append(key)

viral_scores = []
viral_p_values = []
for index, row in ongoing_data.iterrows():
    runs = []
    # calculate the score of viral gene
    sources = interactions[interactions["viral"].str.lower() == row["viral"]]["entrezgene"].to_list()
    score = sum([sum(v[target_index]) for k, v in gene_scores.items() if k in sources]) / len(sources)
    runs.append(score)
    viral_scores.append(score)

    # generate 100 random scores
    for i in range(100):
        random_sources = []
        for s in sources:
            random_sources.append(random.sample(population=genes_by_degree[g.degree(s)], k=1)[0])
        score = sum([sum(v[target_index]) for k, v in gene_scores.items() if k in random_sources]) / len(sources)
        runs.append(score)

    pvalue = (102 - sp.stats.rankdata(runs, method="ordinal")[0]) / 101
    viral_p_values.append(pvalue)

ongoing_data["pvalue_random_source_same_degree"] = viral_p_values
ongoing_data["adjusted_pvalue_random_source_same_degree"] = sm.stats.multipletests(
    ongoing_data["pvalue_random_source_same_degree"], alpha=0.05, method="fdr_bh", is_sorted=False)[1]
# %%
# calculate score by target
for tar_id, name in zip(target, target_names):
    target_score = []
    for index, row in ongoing_data.iterrows():
        # calculate the score of viral gene
        sources = interactions[interactions["viral"].str.lower() == row["viral"]]["entrezgene"].to_list()
        score = sum([v[gene_indexes[tar_id]] for k, v in gene_scores.items() if k in sources]) / len(sources)
        target_score.append(score)
    ongoing_data[name + "_score"] = target_score

# %%
# Plots

# correlation of number of interactions
correlation_1 = round(sp.stats.pearsonr(ongoing_data["interactions_count"], ongoing_data["perm_3_days"])[0], 4)
fig = px.scatter(ongoing_data, x="interactions_count", y="perm_3_days", text="viral")
fig.update_traces(textposition="top center")
fig.update_layout(height=800, title_text=f"""num_interaction Vs Permeability, Pearson={correlation_1}""")
fig.show()

# correlation of number of average score
correlation_2 = round(sp.stats.pearsonr(ongoing_data["network_score"], ongoing_data["perm_3_days"])[0], 4)
fig = px.scatter(ongoing_data, x="network_score", y="perm_3_days", text="viral")
fig.update_traces(textposition="top center")
fig.update_layout(height=800, title_text=f"""Network Score(average) Vs Permeability, Pearson={correlation_2}""")
fig.show()

# correlation of number of adj-pvalue random sources
correlation_4 = round(sp.stats.pearsonr(ongoing_data["adjusted_pvalue"], ongoing_data["perm_3_days"])[0], 4)
fig = px.scatter(ongoing_data, x="adjusted_pvalue", y="perm_3_days", text="viral")
fig.update_traces(textposition="top center")
fig.update_layout(height=800, title_text=f"""Adj(BH) P-value(random sources) Vs Permeability, Pearson={correlation_4}""")
fig.show()

# correlation of number of adj-pvalue random networks
correlation_5 = round(sp.stats.pearsonr(ongoing_data["adjusted_pvalue_multi_net"], ongoing_data["perm_3_days"])[0], 4)
fig = px.scatter(ongoing_data, x="adjusted_pvalue_multi_net", y="perm_3_days", text="viral")
fig.update_traces(textposition="top center")
fig.update_layout(height=800, title_text=f"""Adjusted P-value(multi-net) Vs Permeability, Pearson={correlation_5}""")
fig.show()

# correlation of number of adj-pvalue random sources same degree
correlation_6 = round(sp.stats.pearsonr(ongoing_data["pvalue_random_source_same_degree"], ongoing_data["perm_3_days"])[0], 4,)
fig = px.scatter(ongoing_data, x="pvalue_random_source_same_degree", y="perm_3_days", text="viral")
fig.update_traces(textposition="top center")
fig.update_layout(height=800, title_text=f"""P-value(random source same degree) Vs Permeability, Pearson={correlation_6}""")
fig.show()

# heatmap of effects
viral_protiens = ongoing_data["viral"].tolist()
vascular_protiens = target_names
effects = ongoing_data[[i + "_score" for i in target_names]].to_numpy()
trace = go.Heatmap(x=vascular_protiens, y=viral_protiens, z=effects, type="heatmap", colorscale="Viridis")
data = [trace]
fig = go.Figure(data=data)
fig.update_xaxes(side="top")
fig.update_layout(
    showlegend=False,
    width=1000,
    height=1200,
    autosize=True,
    xaxis_title="Vascular Proteins",
    yaxis_title="Viral Protiens",)
fig.show()

# %%
# heatmap of effects
# viral_protiens = ongoing_data["viral"].tolist()
# vascular_protiens = target_names

# vascular = []
# for index, row in ongoing_data.iterrows():
#     # calculate the score of viral gene
#     sources = interactions[interactions["viral"].str.lower(
#     ) == row["viral"]]["entrezgene"].to_list()
#     score = sum([v[target_index] for k, v in gene_scores.items() if k in sources])/len(sources)
#     vascular.append(score)

# # effects = np.array(vascular)
# effects = ongoing_data[[i + "_score" for i in target_names]].to_numpy()

# trace = go.Heatmap(x=vascular_protiens, y=viral_protiens,
#                    z=effects, type="heatmap", colorscale="Viridis")
# data = [trace]
# fig = go.Figure(data=data)
# fig.update_xaxes(side="top")
# fig.update_layout(
#     showlegend=False,
#     width=1000,
#     height=1200,
#     autosize=True,
#     xaxis_title="Vascular Proteins",
#     yaxis_title="Viral Protiens",
# )
# fig.show()


# %%
#  multiple regression
print('muli-regression permeability')
X = ongoing_data[[i + "_score" for i in target_names]]
Y = ongoing_data["perm_3_days"]
X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
model.summary()

# %%
# permeability normalized regression
print('muli-regression normalized permeability')
X = ongoing_data[[i + "_score" for i in target_names]]
Y = sc.preprocessing.minmax_scale(ongoing_data["perm_3_days"])
X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
model.summary()

# %%
# random regression with 12 targets
print('muli-regression normalized permeability with random 12 genes')
tar_id = random.sample(population=list(g.nodes), k=12)
rand_X = pd.DataFrame(columns=['x'])

for _id in tar_id:
    target_score = []
    for index, row in ongoing_data.iterrows():
        # calculate the score of viral gene
        sources = interactions[interactions["viral"].str.lower() == row["viral"]]["entrezgene"].to_list()
        score = sum([v[gene_indexes[_id]] for k, v in gene_scores.items() if k in sources]) / len(sources)
        target_score.append(score)
    rand_X[_id] = target_score

X = rand_X[[i for i in tar_id]]
Y = sc.preprocessing.minmax_scale(ongoing_data["perm_3_days"])
X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
model.summary()

# %%
# Cateninβ _score regression
print('Cateninβ regression')
X = ongoing_data["Cateninβ  _score"]
Y = sc.preprocessing.minmax_scale(ongoing_data["perm_3_days"])
X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
model.summary()

# %%
# regression with most sig features
print('regression with most sig features')
X = ongoing_data[["Cateninβ  _score", "Cadherin-5_score", 'Cadherin-3_score', 'Catenin delta 1_score']]
Y = sc.preprocessing.minmax_scale(ongoing_data["perm_3_days"])
X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
model.summary()
