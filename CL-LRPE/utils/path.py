from utils import node2vec 
import torch
import networkx as nx
import numpy as np
import multiprocessing as mp
import random
import utils.node2vec
import torch.nn as nn
import torch.nn.functional as F
from networkx.algorithms import bipartite
from numpy import *
import matplotlib.pyplot as plt 
from itertools import chain
# from args import *
from utils.process import *

#Reachability Computation Function
def get_target_random_walks(args, adj):
    r_adj = sparse_mx_to_torch_sparse_tensor(adj).float()

    edge_index = r_adj._indices()

    graph = nx.Graph()
    edge_list = edge_index.transpose(1,0).tolist()
    #print(edge_list)

    graph.add_edges_from(edge_list)

    
    st_node,walks = graph_random_walks(args ,graph)
    
    return walks


def graph_random_walks(args, graph): 

    for edge in graph.edges():
        graph[edge[0]][edge[1]]['weight'] = 1

    G = node2vec.Graph(graph, args.directed, args.p, args.q,args.fastRandomWalk)
    G.preprocess_transition_probs()
    theta=2.0
    st_node, walks = G.simulate_walks(args.num_walks, args.walk_length,theta)

    return st_node, walks
def k_hop_neighborhood(args, adj):
    r_adj = sparse_mx_to_torch_sparse_tensor(adj).float()

    edge_index = r_adj._indices()

    graph = nx.Graph()
    edge_list = edge_index.transpose(1,0).tolist()
    #print(edge_list)

    graph.add_edges_from(edge_list)
    subgraph = []
    for node in list(graph.nodes):
        ngh = get_k_hop_neighborhood(node, 2, graph)
        subgraph.append(ngh)

    return subgraph
def get_k_hop_neighborhood(node, k, graph):
    node_list = []
    ngh = []
    output = {}
    layers = dict(nx.bfs_successors(graph, source=node, depth_limit=k))
    nodes = [node]
    node_list.append(node)
    ngh.append(node)
    for i in range(1, k+1):
        output[i] = []
        for x in nodes:
            output[i].extend(layers.get(x, []))
        nodes = output[i]
        ngh.extend(nodes)
    
    ngh = np.random.choice(ngh, size=29)
    node_list.extend(ngh)
    
    return node_list

#子图序列生成

def choose_next_node(current_node, previous_node, neighbors, graph, p, q):
    # 根据 p 和 q 权重选择下一个节点
    weights = []
    for neighbor in neighbors:
        if neighbor == previous_node:
            weights.append(1 / p)
        elif graph.has_edge(current_node, neighbor):
            weights.append(1)
        else:
            weights.append(1 / q)

    weights = np.array(weights)
    weights /=  weights.sum()

   
    return random.choice(neighbors, p=weights)


def random_walk(start_node, graph, walk_length, p, q):
    walk = [start_node]
    for _ in range(walk_length - 1):
        current_node = walk[-1]
        neighbors = list(graph.neighbors(current_node))

        if len(neighbors) > 0:
            if len(walk) == 1:
                
                next_node = random.choice(neighbors)
            else:
             
                next_node = choose_next_node(current_node, walk[-2], neighbors, graph, p, q)

            walk.append(next_node)
        else:
            break

    return walk

def generate_subgraph_sequence(args, node_sequence, adj):
    r_adj = sparse_mx_to_torch_sparse_tensor(adj).float()

    edge_index = r_adj._indices()

    graph = nx.Graph()
    edge_list = edge_index.transpose(1,0).tolist()

    graph.add_edges_from(edge_list)
    
    for edge in graph.edges():
        graph[edge[0]][edge[1]]['weight'] = 1

    subgraph_sequence = []

    for node in node_sequence:
        subgraph = random_walk(node, graph, args.sugraph_node_number, args.p, args.q)
        subgraph_sequence.append(subgraph)

    return subgraph_sequence

def generate_subgraph_feature_sequence(subgraph_list, feature):
    feature_list = []
    for subgraph in subgraph_list:
        subgraph_feature = feature[subgraph]
        subgraph_feature = torch.flatten(subgraph_feature)
        feature_list.append(subgraph_feature)

    return feature_list



     




