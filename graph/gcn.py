import tensorflow as tf
from tensorflow import keras
import spektral
import numpy as np
import scipy.sparse as sp
import numpy as np
import time

#load dataset
# interaction matrix

dataset_obj = spektral.datasets.citation.Citation('cora', random_split=False, normalize_x=False, dtype=np.float32)
cora_graph = dataset_obj[0]
adj = cora_graph.a # adjacency matrix
node_feat = cora_graph.x # node features
edge_feat = cora_graph.e # edge features
labels = cora_graph.y # node labels


print("Number of nodes: ", cora_graph.n_nodes)
print("Number of edges: ", cora_graph.n_edges)
print("adj matrix data type: ", type(adj)) #csr_array
print("node_feats data type: ", type(node_feat), node_feat.shape)
print("Node Num:1' features", node_feat[0], labels[0])

print("train_dataset", dataset_obj.mask_tr)
print("validation_dataset", dataset_obj.mask_va)
print("test_dataset", dataset_obj.mask_te)

# train_dataset = dataset_obj.mask_tr[..., np.newaxis] * adj
# val_dataset = dataset_obj.mask_va * adj
# test_dataset = dataset_obj.mask_te * adj
#
# print(train_dataset)

node_degrees = 1 / np.sqrt(adj.sum(axis=0))
diag_matrix = sp.coo_matrix((node_degrees, (np.arange(cora_graph.n_nodes), np.arange(cora_graph.n_nodes))))
Laplacian = diag_matrix @  adj @ diag_matrix.transpose()


