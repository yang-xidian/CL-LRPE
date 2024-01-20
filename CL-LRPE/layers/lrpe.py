import torch
from torch.nn.utils.rnn import  pad_packed_sequence,pad_sequence, pack_padded_sequence
from utils import process, path, args
import numpy as np
import torch
from torch import nn
from layers import * 
from layers.fnn import MLP
import torch_geometric as tg
device = torch.device('cuda:1')

class LRPE(nn.Module):
    def __init__(self, n_in, subg_feature_dim, hidden_dim, mlp_output_dim, hidden_layer, dropout):
        super(LRPE, self).__init__()
        self.subg_feature_dim=subg_feature_dim
        self.hidden_dim = hidden_dim
        self.mlp_output_dim = mlp_output_dim
        self.hidden_layer = hidden_layer
        self.dropout_rate = dropout 

        self.subgraph_feature_downscaling = MLP(num_layers=3, input_dim=self.subg_feature_dim, hidden_dim=self.hidden_dim, output_dim=self.mlp_output_dim)

        self.pattern_extraction = torch.nn.LSTM(
            input_size = n_in,
            hidden_size = self.hidden_dim,
            num_layers = self.hidden_layer,
            dropout = self.dropout_rate,
            bidirectional = True
        )

        self.lin_pattern_h = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.fea_decoder = FNN(self.mlp_output_dim, self.hidden_dim, self.subg_feature_dim, 3)
        self.fea_decoder_loss = nn.MSELoss()
    
    #def forward(self, features, all_subgraph_list, subgraph_fea, tmp):
    def forward(self, features, subgraph_fea, tmp):
        '''self.all_subgraph_feature_list = []
        self.subg_length = [] #记录一个序列中子图的个数

        for subgraph_list in all_subgraph_list:
            
            subgraph_feature = path.generate_subgraph_feature_sequence(subgraph_list, features)
            self.all_subgraph_feature_list.append(subgraph_feature)
            self.subg_length.append(len(subgraph_list))
        
        all_fea=torch.tensor([sf.cpu().detach().numpy() for subgraphlist in self.all_subgraph_feature_list for sf in subgraphlist ]).to(device)

        subgraph_feature_downscaling = self.subgraph_feature_downscaling(all_fea)

        subgraph_feature_decoder_loss = self.fea_decoder_loss(all_fea, self.fea_decoder(subgraph_feature_downscaling))
        
        
        subgraph_feature_downscaling = torch.chunk(subgraph_feature_downscaling, chunks=len(self.all_subgraph_feature_list), dim=0)
        subgraph_feature_downscaling = torch.stack(subgraph_feature_downscaling, dim=0)
        '''
        all_subgraph_fea = []
        self.subg_length = [] #记录一个序列中子图的个数
        for t in tmp:
            all_subgraph_fea.append(subgraph_fea[t])
            self.subg_length.append(len(t))
        subgraph_feature = torch.tensor([sub.cpu().detach().numpy() for sub in all_subgraph_fea ]).to(device)
        
        arr_pack = pack_padded_sequence(subgraph_feature, self.subg_length, batch_first=True)
        pattern_information = self.pattern_extraction(arr_pack)

        pattern_information_unpack = pad_packed_sequence(pattern_information[0], batch_first=True)
        
        node_pattern_information = []
        for v in pattern_information_unpack[0]:  #这里用的是简单做法，把每个序列中的第0个向量直接作为该目标节点的图案特征
            node_pattern_information.append(v[0])

        node_pattern_information = torch.stack(node_pattern_information, dim=0)
        

        node_pattern_information = self.lin_pattern_h(node_pattern_information)


        



        return node_pattern_information
    
