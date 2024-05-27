import torch
import torch.nn as nn
from layers import GCN, AvgReadout, Discriminator, LSTMContext, FNN
import pdb
import torch_geometric as tg
from layers.sgc import SGC
from layers.fnn import *
import numpy as np
from utils import args
from layers.fnn import MLP

device = torch.device('cuda:1')
args = args.make_args()

class MOST(nn.Module):
    def __init__(self, n_in, n_h, activation, nb_classes):
        super(MOST, self).__init__()
        # self.gcn = GCN(n_in, n_h, activation)
        # self.gcn = LinearModel(n_in, n_h)
        # self.gcn = tg.nn.GATConv(n_in, n_h)
        # self.gcn = GAT(n_in, n_h, activation)


        self.gcn = LSTMContext(obj_classes=7, hidden_dim=n_h, nhidlayer=2, nl_edge=2, dropout=0.4, in_channels=n_in)
        # self.gcn = LSTMContext(obj_classes=7, hidden_dim=n_h, nhidlayer=2, nl_edge=1, dropout=0.4, in_channels=n_in)
        self.MLP = MLP(num_layers=4, input_dim=n_h, hidden_dim=int(n_h/2), output_dim=nb_classes)

        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()




  
    def forward(self, seq1, subgraph, adj, sparse, msk, samp_bias1, samp_bias2):
        h_1_l1, h_1_l2, c_out = self.gcn(seq1, subgraph, adj) 

        output = self.MLP(h_1_l2)

    
       
        return output
    
