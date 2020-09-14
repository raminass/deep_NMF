# %% [markdown]
## Refs
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
import plotly.graph_objects as go
import mygene
import random
import joblib

mg = mygene.MyGeneInfo()

# %%
# Global Variables
PROPAGATE_ALPHA = 0.9
PROPAGATE_ITERATIONS = 200
PROPAGATE_EPSILON = 10 ** (-5)
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

# %%
def read_network(network_filename):
    network = pd.read_table(network_filename, header=None, usecols=[0, 1, 2])
    return nx.from_pandas_edgelist(network, 0, 1, 2)


def generate_similarity_matrix(network):
    genes = sorted(network.nodes)
    matrix = nx.to_scipy_sparse_matrix(network, genes, weight=2)

    norm_matrix = sp.sparse.diags(1 / sp.sqrt(matrix.sum(0).A1))
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


def generate_propagate_data(network):
    matrix, genes = generate_similarity_matrix(network)
    num_genes = len(genes)
    gene_indexes = dict([(gene, index) for (index, gene) in enumerate(genes)])
    gene_scores = {gene: propagate([gene], matrix, gene_indexes, num_genes) for gene in genes}

    return matrix, num_genes, gene_indexes, gene_scores


# %%
# Loading Data
g = read_network("H_sapiens.net")
interactions = pd.read_csv("interactions.csv")  # https://www.nature.com/articles/s41586-020-2286-9#Sec36
perm3 = pd.read_csv("perm3.csv")  # permeability after 3 Days
perm4 = pd.read_csv("perm4.csv")  # permeability after 4 Days

# %%
# propogate Network
W, num_genes, gene_indexes, gene_scores = generate_propagate_data(g)
# target_index = [gene_indexes[i] for i in target if i in gene_indexes]

# network_scores = {"W": W, "num_genes": num_genes, "gene_indexes": gene_indexes, "gene_scores": gene_scores}
# save scores artifact
# joblib.dump(network_scores, "network_scores.pkl")

# load scores artifact
# network_scores = joblib.load("network_scores.pkl")

# %%
# mapping to entrez id
xli = interactions["PreyGene"].unique().tolist()
out = pd.DataFrame(mg.querymany(xli, scopes="symbol", fields="entrezgene", species="human"))
interactions = pd.merge(interactions, out[["query", "entrezgene"]], left_on="PreyGene", right_on="query")
interactions["entrezgene"] = interactions["entrezgene"].astype(int)

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
correlation = round(sp.stats.pearsonr(corr_data["PreyGene"], corr_data["Permeability Value (3 days)"])[0], 4,)

# %%
# plotly figure setup
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        name="X vs Y",
        x=corr_data["PreyGene"],
        y=corr_data["Permeability Value (3 days)"],
        text=corr_data["viral"].to_list(),
        mode="markers+text",
    )
)
# fig.add_trace(go.Scatter(name='line of best fit', x=X, y=df['bestfit'], mode='lines'))
fig.update_traces(textposition="top center")
# plotly figure layout
fig.update_layout(
    height=600,
    width=1000,
    xaxis_title="Human interaction count",
    yaxis_title="Permeability Value (3 days)",
    annotations=[
        go.layout.Annotation(
            text=f"Pearson = {correlation}",
            align="left",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.9,
            y=0.9,
            bordercolor="red",
            borderwidth=1,
        )
    ],
)
fig.show()


# %%
# calculate p-value and scores

viral_scores = []
viral_p_values = []
for index, row in corr_data.iterrows():
    runs = []
    # calculate the score of viral gene
    sources = interactions[interactions["viral"].str.lower() == row["viral"]]["entrezgene"].to_list()
    score = sum([sum(v[target_index]) for k, v in gene_scores.items() if k in sources])
    runs.append(score)
    viral_scores.append(score)

    # generate 100 random scores
    for i in range(100):
        random_sources = random.sample(population=list(gene_indexes.keys()), k=len(sources))
        score = sum([sum(v[target_index]) for k, v in gene_scores.items() if k in random_sources])
        runs.append(score)

    pvalue = (101 - sp.stats.rankdata(runs, method="ordinal")[0]) / 101
    viral_p_values.append(pvalue)

corr_data["network_scores"] = viral_scores
corr_data["pvalue"] = viral_p_values

# %%
# Plot network score correlation

# corr_data[corr_data['viral'].isin(['nsp7','nsp13'])]["network_score"]
correlation_3 = round(sp.stats.pearsonr(corr_data["pvalue"], corr_data["Permeability Value (3 days)"])[0], 4,)
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        name="X vs Y",
        x=corr_data["pvalue"],
        y=corr_data["Permeability Value (3 days)"],
        text=corr_data["viral"].to_list(),
        mode="markers+text",
    )
)
# fig.add_trace(go.Scatter(name='line of best fit', x=X, y=df['bestfit'], mode='lines'))
fig.update_traces(textposition="top center")
# plotly figure layout
fig.update_layout(
    height=600,
    width=1000,
    xaxis_title="pvalue",
    yaxis_title="Permeability Value (3 days)",
    annotations=[
        go.layout.Annotation(
            text=f"Pearson = {correlation_3}",
            align="left",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.9,
            y=0.9,
            bordercolor="red",
            borderwidth=1,
        )
    ],
)
fig.show()

# %%
