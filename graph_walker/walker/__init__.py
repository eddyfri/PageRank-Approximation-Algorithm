import time

import networkx as nx
import numpy as np
from _walker import node2vec_random_walks as _node2vec_random_walks
from _walker import random_walks as _random_walks
from _walker import random_walks_with_restart as _random_walks_with_restart
from _walker import weighted_corrupt as _corrupt

from .preprocessing import get_normalized_adjacency
from .preprocessing import get_normalized_adjacency_list


def random_walks(
    G,
    n_walks=10,
    walk_len=10,
    sub_sampling=0.,
    p=1, q=1, alpha=0,
    start_nodes=None,
    verbose=True
):
    start_time = time.time()

    debug_rw = False

    #A = get_normalized_adjacency(G, sub_sampling=sub_sampling)
    #indptr = A.indptr.astype(np.uint32)
    #indices = A.indices.astype(np.uint32)
    #data = A.data.astype(np.float32)
    adj_lists = get_normalized_adjacency_list(G)
    nodes_list = [int(n) for n in G]
    max_node_id = max(nodes_list)
    num_edges = G.size()
    if debug_rw:
        print("nodes_list",nodes_list)
        print("max_node_id",max_node_id)

    indptr = np.zeros(max_node_id+1,dtype=np.uint32)
    outdegrees = np.zeros(max_node_id+1,dtype=np.uint32)
    indices = np.zeros(num_edges,dtype=np.uint32)
    data = np.zeros(num_edges,dtype=np.float32)

    if debug_rw:
        print(nodes_list)
        print(adj_lists)
    j = 0
    for adj_list in adj_lists:
        node = adj_list[0]
        neighbors = adj_list[1]
        neighbors = [int(i) for i in neighbors.keys()]
        if debug_rw:
            print("adj_list of node ",node,"is",neighbors)

        indptr[node] = j
        outdegrees[node] = len(neighbors)
        weight_node_ = 0.
        if len(neighbors) > 0:
            weight_node_ = 1./float(len(neighbors))
        else:
            indptr[node] = num_edges+1
        for node_neighbor in neighbors:
            indices[j] = node_neighbor
            data[j] = weight_node_
            j = j+1

    if debug_rw:
        print("indptr",indptr)
        print("outdegrees",outdegrees)
        print("indices",indices)
        print("data",data)

    if start_nodes is None:
        #start_nodes = np.arange(len(G.nodes)).astype(np.uint32)
        start_nodes = np.array(nodes_list, dtype=np.uint32)
    else:
        start_nodes = np.array(start_nodes, dtype=np.uint32)

    if p == 1 and q == 1:
        if alpha == 0:
            walks = _random_walks(
                indptr, indices, data, start_nodes,
                n_walks, walk_len)
        else:
            walks = _random_walks_with_restart(indptr, outdegrees, indices, data, start_nodes, max_node_id, num_edges, n_walks, alpha)
    else:
        walks = _node2vec_random_walks(
            indptr, indices, data, start_nodes,
            n_walks, walk_len, p, q)

    if verbose:
        duration = time.time() - start_time
        print(f"Random walks - T={duration:.2f}s")
    return walks


def corrupt(
    G, walks, r=.01, ns_exponent=.75,
    negative_size=100000, verbose=True
):
    # corrupt random walks
    start_time = time.time()

    n_nodes = len(G.nodes)
    A = nx.adjacency_matrix(G)
    indptr = A.indptr.astype(np.uint32)
    indices = A.indices.astype(np.uint32)

    # compute weights for each node
    weights = np.array(
        [G.degree(node, "weight") for node in G.nodes],
        dtype=np.float32)
    weights **= ns_exponent
    weights /= weights.sum()

    # draw negative table
    neg = np.random.choice(
        range(n_nodes),
        size=negative_size,
        p=weights,
        replace=True)

    # corrupt random walks
    similarity = _corrupt(walks, neg, n_nodes, r)

    if verbose:
        elapsed = time.time() - start_time
        print(f"Corrupt random walks - T={elapsed:.02}s")

    return similarity


def corrupted_random_walks(
    G,
    n_walks=10,
    walk_len=10,
    sub_sampling=0.,
    p=1, q=1, r=.1,
    ns_exponent=.75,
    negative_size=100000,
    start_nodes=None,
    verbose=True
):
    start_time = time.time()

    n_nodes = len(G.nodes)
    A = get_normalized_adjacency(G, sub_sampling=sub_sampling)
    indptr = A.indptr.astype(np.uint32)
    indices = A.indices.astype(np.uint32)
    data = A.data.astype(np.float32)

    if start_nodes is None:
        start_nodes = np.arange(len(G.nodes)).astype(np.uint32)
    else:
        start_nodes = np.array(start_nodes, dtype=np.uint32)

    if p == 1 and q == 1:
        walks = _random_walks(
            indptr, indices, data, start_nodes,
            n_walks, walk_len)
    else:
        walks = _node2vec_random_walks(
            indptr, indices, data, start_nodes,
            n_walks, walk_len, p, q)

    if verbose:
        duration = time.time() - start_time
        print(f"Random walks - T={duration:.2f}s")

    # corrupt random walks
    start_time = time.time()

    # compute weights for each node
    weights = np.array(
        [G.degree(node, "weight") for node in G.nodes],
        dtype=np.float32)
    weights **= ns_exponent
    weights /= weights.sum()

    # draw negative table
    neg = np.random.choice(
        range(n_nodes),
        size=negative_size,
        p=weights,
        replace=True)

    # corrupt random walks
    similarity = _corrupt(walks, neg, n_nodes, r)

    if verbose:
        elapsed = time.time() - start_time
        print(f"Corrupt random walks - T={elapsed:.02}s")

    return walks, similarity
