import torch
import torch.nn as nn
from layers import GCN, AvgReadout, Discriminator, LSTMContext, LinearModel, FNN
import pdb
import torch_geometric as tg
from layers.sgc import SGC
from layers.fnn import *
import numpy as np
from utils import args
from layers.lrpe import LRPE


device = torch.device('cuda:1')
args = args.make_args()

class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation, subgraph):
        super(DGI, self).__init__()
        # self.gcn = GCN(n_in, n_h, activation)
        # self.gcn = LinearModel(n_in, n_h)
        # self.gcn = tg.nn.GATConv(n_in, n_h)
        # self.gcn = GAT(n_in, n_h, activation)

        self.subgraph = subgraph

        self.LSTM = LSTMContext(obj_classes=7, hidden_dim=n_h, nhidlayer=2, nl_edge=2, dropout=0.4, in_channels=n_in)
        # self.gcn = LSTMContext(obj_classes=7, hidden_dim=n_h, nhidlayer=2, nl_edge=1, dropout=0.4, in_channels=n_in)

        self.lrpe = LRPE(n_in=n_in, subg_feature_dim=n_in*args.sugraph_node_number, hidden_dim=n_h, mlp_output_dim=args.subgraph_feature_downscaling_dim, hidden_layer=2, dropout=0.4)
        self.gcn = GCN(n_in, n_in, activation)
        self.gcn2 = GCN(n_in, n_in, activation)
        

        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

        self.sfea = nn.Linear(n_in, n_h)

        self.lfea = nn.Linear(n_h*2, n_h)

        self.feature_decoder = FNN(n_h, n_h, n_in, 3)
        #self.feature_decoder = MLPWithSelfAttention(n_h, n_h, n_in, n_h)
        self.feature_loss_func = nn.MSELoss()

        self.feature2_decoder = FNN(n_h, n_h, n_in, 3)
        #self.feature2_decoder = MLPWithSelfAttention(n_h, n_h, n_in, n_h)
        self.feature2_loss_func = nn.MSELoss()

        #self.feature_decoder3 = FNN(n_h, n_h, n_in, 3)
        self.feature3_loss_func = nn.MSELoss()

        self.localsim_decoder = FNN(n_h, n_h, args.walk_length, 3)

        self.feature4_loss_func = nn.MSELoss()
        self.subgraph_fea_decoder = FNN(n_h, n_h, n_in, 3)

        self.fea_agg = MLP(num_layers=3, input_dim=n_h*2 + n_in, hidden_dim=n_h, output_dim=n_h)
        #self.fea_agg = MLP(num_layers=3, input_dim=n_h + n_in, hidden_dim=n_h, output_dim=n_h)

        self.weight = torch.nn.Parameter(torch.Tensor(n_h, n_h))

    def forward2(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.LSTM(seq1, adj, sparse) # (1, 2708, 512) -> (2708, 512)

        # h_1, c_out = self.gcn(seq1, self.subgraph, adj)
        # h_1 = self.gcn(seq1, adj._indices())

        c = self.read(h_1, msk) # (1, 512) #(512)
        c = self.sigm(c)
        h_2 = self.LSTM(seq2, adj, sparse) # (2708, 512)
        # h_2, _ = self.gcn(seq2, self.subgraph, adj)
        # h_2 = self.gcn(seq2, adj._indices())
        # print(c.shape)
        # print(c_out.shape)
        # pdb.set_trace()
        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)
        # ret = self.disc(c_out, h_1, h_2, samp_bias1, samp_bias2)

        return ret
    
    #没有将两层decoder都放入的情况 
    def forward_initial(self, seq1, subgraph2, adj, sparse, msk, samp_bias1, samp_bias2):
        # h_1 = self.gcn(seq1, adj, sparse) # (1, 2708, 512) -> (2708, 512)
        h_1, c_out = self.LSTM(seq1, self.subgraph, adj)
        # h_1 = self.gcn(seq1, adj._indices())

        c = self.read(h_1, msk) # (1, 512) #(512)
        c = self.sigm(c)
        # h_2 = self.gcn(seq2, adj, sparse) # (2708, 512)
        h_2, _ = self.LSTM(seq1, subgraph2, adj)
        # h_2 = self.gcn(seq2, adj._indices())
        # print(c.shape)
        # print(c_out.shape)
        # pdb.set_trace()
        # ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)
        ret = self.disc(c_out, h_1, h_2, samp_bias1, samp_bias2) 

        # Decoder 部分
        feature_loss = self.feature_loss_func(seq1, self.feature_decoder(h_1)) 

        return ret, feature_loss
    
    def discriminator(self, z, s_z):
        value = torch.matmul(z, torch.matmul(self.weight, s_z))
        return torch.sigmoid(value)
    
    
    #def forward(self, seq1, neg, all_subgraph_list, tmp, adj, sparse, msk, samp_bias1, samp_bias2):
    def forward(self, seq1, neg, tmp, adj, sparse, msk, samp_bias1, samp_bias2):
        h_1_l1, h_1_l2, c_out = self.LSTM(seq1, self.subgraph, adj) 
        
        fea_subgraph = self.gcn(seq1, adj, True)
        fea_subgraph = self.gcn2(fea_subgraph, adj, True)
        
        #pattern_information = self.lrpe(seq1, all_subgraph_list, fea_subgraph, tmp)
        pattern_information = self.lrpe(seq1, fea_subgraph, tmp)
                                                                 
        c = self.read(h_1_l1, msk) # (1, 512) #(512)
        c = self.sigm(c)

        #h_2, _, _ = self.gcn(seq1, subgraph2, adj)

        h_neg, _, _ = self.LSTM(seq1, neg, adj)
        

        #mi_loss = - torch.log(torch.mul(h_1_l1, h_2).mean(dim=1) / (torch.norm(h_1_l1, dim=1) + torch.norm(h_2, dim=1)) + 1e-7).sum()

        feaid = [i[0] for i in self.subgraph]
        #new_input = torch.cat((h_1_l1, h_2), 1)
        #new_input = torch.cat((new_input, seq1[feaid]), 1)

        new_input = torch.cat((h_1_l1, seq1[feaid]), 1)
        
        new_input = torch.cat((new_input, pattern_information), 1)
    
        fea = self.fea_agg(new_input)
        
        ret = self.disc(c_out, h_1_l1, h_neg, samp_bias1, samp_bias2) 

        # return ret, None
        #sf = self.sfea(seq1)
        #h1, h2, h3, h4, h = self.sgc(seq1, adj)
        #hidden_feature = torch.cat((h1, h2), 1)
        #hidden_feature = torch.cat((hidden_feature, h3), 1)
        #hidden_feature = torch.cat((hidden_feature, h4), 1)
        #hidden_feature = torch.cat((h, h_1_l2), 1)
        #hidden_feature = torch.cat((hidden_feature, sf), 1)

        #lf = self.lfea(hidden_feature)



        # Decoder 部分

        cosine_similarities = torch.mm(seq1, seq1.T)

        neighbor_sim = [cosine_similarities[i][self.subgraph[i]] for i in range(len(self.subgraph))]
        neighbor_sim = torch.tensor([item.cpu().detach().numpy() for item in neighbor_sim]).to(device)

        #fea_decoder = self.feature_decoder3(h_1_l1)
        #cosine_similarities_decoder = torch.mm(fea_decoder, fea_decoder.T)
        #neighbor_sim_decoder = [cosine_similarities_decoder[i][self.subgraph[i]] for i in range(len(self.subgraph))]
        #neighbor_sim_decoder = torch.tensor([item.cpu().detach().numpy() for item in neighbor_sim_decoder]).to(device)
        
        neighbor_sim_decoder = self.localsim_decoder(h_1_l1)
        feature_loss3 = self.feature3_loss_func(neighbor_sim, neighbor_sim_decoder)

        #feature_loss = self.feature_loss_func(seq1, fea_decoder) # feature层面的decoder
        feature_loss = self.feature_loss_func(seq1, self.feature_decoder(h_1_l1)) # feature层面的decoder
        #feature_loss = self.feature_loss_func(seq1, self.feature_decoder(h_1_l2)) # feature层面的decoder

        feature_loss2 = self.feature2_loss_func(seq1, self.feature2_decoder(fea))
        feature_loss4 = self.feature4_loss_func(seq1, self.subgraph_fea_decoder(pattern_information))

        print(feature_loss, feature_loss2, feature_loss3, feature_loss4)

        #feature_loss2 = self.feature2_loss_func(seq1, self.feature2_decoder(h_1_l2))

        #feature_loss3 = self.feature3_loss_func(seq1, self.feature3_decoder(lf))
        

        #return feature_loss
        #return feature_loss + feature_loss2 + feature_loss3
        #return  ret, feature_loss + 0.0000001*feature_loss3
        return ret, feature_loss + feature_loss2 + 0.0000001*feature_loss3 
        #return ret, feature_loss + feature_loss2 + 0.0000001*feature_loss3 + 1e-15*mi_loss
        #return ret,  feature_loss + 0.0000001*feature_loss3
        #return ret, feature_loss + feature_loss2 + 0.0000001*feature_loss3 + 1e-15*mi_loss
        #return feature_loss + feature_loss2 + 0.0000001*feature_loss3
        #return ret, feature_loss2
        #return ret, feature_loss * 0.9 + feature_loss2 * 0.1
        #return ret, feature_loss

    # Detach the return variables
    def embed(self, seq, tmp, adj, sparse, msk):
        # h_1 = self.gcn(seq, adj, sparse)
        h_1, h_2, _ = self.LSTM(seq, self.subgraph, adj)
        #_, h_1, _ = self.gcn(seq, self.subgraph, adj)
        # h_1 = self.gcn(seq, adj._indices())
        #sf = self.sfea(seq)
        fea_subgraph = self.gcn(seq, adj, True)

        #pattern_information = self.lrpe(seq, all_subgraph_list, fea_subgraph, tmp)
        pattern_information = self.lrpe(seq, fea_subgraph, tmp)
        #h_2_1, _, _ = self.gcn(seq, subgraph2, adj)

        
        feaid = [i[0] for i in self.subgraph]
        #new_input = torch.cat((h_1, h_2_1), 1)
        new_input = torch.cat((h_1, seq[feaid]), 1)
        new_input = torch.cat((new_input, pattern_information), 1)
        
        fea = self.fea_agg(new_input)
        
        #hidden_feature = torch.cat((h1, h2), 1)
        #hidden_feature = torch.cat((hidden_feature, h3), 1)
        #hidden_feature = torch.cat((hidden_feature, h4), 1)
        #hidden_feature = torch.cat((h, h_1), 1)
        #hidden_feature = torch.cat((hidden_feature, sf), 1)

        #lf = self.lfea(hidden_feature)


        c = self.read(h_1, msk)

        return fea.detach(), c.detach()

class DGI2(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGI2, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)
        # self.gcn = LinearModel(n_in, n_h)
        # self.gcn = tg.nn.GATConv(n_in, n_h)
        # self.gcn = GAT(n_in, n_h, activation)

        # self.gcn = LSTMContext(obj_classes=7, hidden_dim=n_h, nhidlayer=2, nl_edge=2, dropout=0.5, in_channels=n_in)

        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

    def forward(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn(seq1, adj, sparse) # (1, 2708, 512) -> (2708, 512)
        # h_1 = self.gcn(seq1, self.subgraph, adj)
        # h_1 = self.gcn(seq1, adj._indices())

        c = self.read(h_1, msk) # (1, 512) #(512)
        c = self.sigm(c)
        h_2 = self.gcn(seq2, adj, sparse) # (2708, 512)
        # h_2 = self.gcn(seq2, self.subgraph, adj)
        # h_2 = self.gcn(seq2, adj._indices())

        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

        return ret

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk):
        h_1 = self.gcn(seq, adj, sparse)
        # h_1 = self.gcn(seq, self.subgraph, adj)
        # h_1 = self.gcn(seq, adj._indices())

        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()

