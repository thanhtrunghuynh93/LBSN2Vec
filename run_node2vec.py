import networkx as nx
from node2vec import Node2Vec


import argparse
from scipy.io import loadmat
import os
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="LBSN configurations")
    parser.add_argument('--dataset_name', type=str, default="")
    args = parser.parse_args()
    return args

args = parse_args()
dataset_path = "edgelist_graph/{}.edgelist".format(args.dataset_name)

# Create a graph
graph = nx.read_edgelist(dataset_path)

# Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
node2vec = Node2Vec(graph, dimensions=128, walk_length=80, num_walks=40, workers=16)  # Use temp_folder for big graphs

# Embed nodes
model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)

# Save embeddings for later use
model.wv.save_word2vec_format("node2vec_emb/{}.embedding".format(args.dataset_name))
