import torch
from torch.nn.utils.rnn import  pad_packed_sequence,pad_sequence, pack_padded_sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from layers import * 
from layers.fnn import MLP
import torch_geometric as tg


class LSTMContext(nn.Module):
    # def __init__(self,  obj_classes, hidden_dim, nhidlayer, nl_edge, dropout, in_channels, device):
    def __init__(self,  obj_classes, hidden_dim, nhidlayer, nl_edge, dropout, in_channels):
        super(LSTMContext, self).__init__()
        self.obj_classes = obj_classes
        self.obj_dim = in_channels
        self.hidden_dim = hidden_dim
        self.hidden_layer = nhidlayer
        self.dropout_rate = dropout
        self.nl_edge = nl_edge
        # self.device = device
        self.fea_a = MLP(num_layers=3, input_dim=in_channels + self.hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim)
        activation = F.relu
        withbn = False
        withloop = False

        self.obj_ctx_rnn = torch.nn.LSTM(            
            input_size = self.obj_dim,
            # input_size = self.hidden_dim,
            hidden_size = self.hidden_dim,
            num_layers = self.hidden_layer,
            dropout = self.dropout_rate,
            bidirectional = True
        )

        self.decoder_rnn = torch.nn.LSTM(
            input_size = self.hidden_dim + self.obj_dim,
            # input_size = self.hidden_dim + self.hidden_dim,
            # input_size = self.hidden_dim,
            hidden_size = self.hidden_dim,
            num_layers = self.hidden_layer,
            dropout = self.dropout_rate,
            bidirectional = False
        )

        self.edge_ctx_rnn = torch.nn.LSTM(
            input_size = self.obj_dim + self.hidden_dim,
            hidden_size = self.hidden_dim,
            num_layers = self.nl_edge,
            dropout = self.dropout_rate,
            bidirectional = True
        )

        self.lin_obj_h = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.lin_edge_h = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        self.disc = Discriminator(self.hidden_dim)

        self.act = nn.PReLU()
        self.act2 = nn.PReLU()
        #self.gnn = tg.nn.GATConv(self.hidden_dim, self.hidden_dim)
        self.gnn = tg.nn.GCNConv(in_channels, self.hidden_dim)

        self.gnn2 = tg.nn.GCNConv(self.hidden_dim, self.hidden_dim)

        # self.out_layer = Dense(self.hidden_dim, obj_classes)
        self.lin1 = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.lin2 = nn.Linear(self.hidden_dim, self.hidden_dim)

    def obj_ctx2(self, obj_feas, subgraph):
    # def obj_ctx(self, obj_feas, original_fea, subgraph):
        #subgraph: 长度为7600的一个list: [72, 29, 36,....]
        perm = sorted(range(len(subgraph)), key=lambda i:len(subgraph[i]), reverse=True)
        subgraph.sort(key=lambda x:len(x), reverse=True)

        # new_input = torch.cat((obj_feas, original_fea), 1) 
        # new_input = obj_feas 

        self.length = []
        self.arr = []
        for val in subgraph:
            self.length.append(len(val))
            self.arr.append(obj_feas[val]) #self.arr
            # self.arr.append(new_input[val]) #self.arr
        
        max_length = max(self.length) 

        arr_pad = pad_sequence(self.arr, batch_first=True) 
        
        arr_pack = pack_padded_sequence(arr_pad, self.length, batch_first=True) 
        output = self.decoder_rnn(arr_pack)
        output_unpack = pad_packed_sequence(output[0], batch_first=True) 
        
        obj_new_feature = []
        for v in output_unpack[0]: 
            obj_new_feature.append(v[0])
        
        obj_new_feature = torch.stack(obj_new_feature, dim=0)

        return obj_new_feature

        # return obj_new_feature 
        
        new_input = torch.cat((obj_new_feature, obj_feas), 1) 
        
        # new_input = torch.cat((obj_new_feature, original_fea), 1)

        self.arr = []
        for val in subgraph: 
            self.arr.append(new_input[val])
        
        arr_new_pad = pad_sequence(self.arr, batch_first=True)
        arr_new_pack = pack_padded_sequence(arr_new_pad, self.length, batch_first=True)
        output = self.decoder_rnn(arr_new_pack)

        output_unpack = pad_packed_sequence(output[0], batch_first=True) 
        output_lstm = []
        
        for v in output_unpack[0]: 
            output_lstm.append(v[0])

        output_lstm = torch.stack(output_lstm)

        # c_out = torch.sum(output_unpack[0], 1)

        # print(output_lstm.shape)
        # c_out = torch.sum(output_lstm, 0)
        # print(c_out.shape)

        # pdb.set_trace()

        # return output_lstm, c_out 
        return output_lstm

    
    def obj_ctx(self, obj_feas, subgraph):
    # def obj_ctx(self, obj_feas, original_fea, subgraph):
        #subgraph: 长度为7600的一个list: [72, 29, 36,....]
        
        # perm = sorted(range(len(subgraph)), key=lambda i:len(subgraph[i]), reverse=True)
        # subgraph.sort(key=lambda x:len(x), reverse=True)

        # new_input = torch.cat((obj_feas, original_fea), 1) 
        # new_input = obj_feas

        self.length = [] 
        self.arr = []
        for val in subgraph:
            self.length.append(len(val))
            self.arr.append(obj_feas[val])
            # self.arr.append(new_input[val])
        
        max_length = max(self.length) 
        

        arr_pad = pad_sequence(self.arr, batch_first=True) 
        
        arr_pack = pack_padded_sequence(arr_pad, self.length, batch_first=True) 
        

        output = self.obj_ctx_rnn(arr_pack)  #双向LSTM
        output_unpack = pad_packed_sequence(output[0], batch_first=True) 

        obj_new_feature = []
        for v in output_unpack[0]: 
            obj_new_feature.append(v[0])

        
        obj_new_feature = torch.stack(obj_new_feature, dim=0) 
        obj_new_feature = self.lin_obj_h(obj_new_feature) 
        

        # obj_new_feature = self.lin1(obj_new_feature)
        # obj_new_feature = self.lin2(obj_new_feature)

        # c_out = torch.sum(output_unpack[0], 1) 
        c_out = torch.mean(output_unpack[0], 1)
        c_out = self.lin_edge_h(c_out)
        # print(c_out.shape)
        # pdb.set_trace()

        # return obj_new_feature, c_out 
        #return obj_new_feature, None, c_out

        # return obj_new_feature 
        
        feaid = [i[0] for i in subgraph]
        new_input = torch.cat((obj_new_feature, obj_feas[feaid]), 1) 
        
        node_hidden_feature = self.fea_a(new_input)


        # new_input = torch.cat((obj_new_feature, original_fea), 1)

        '''self.arr = []
        for val in subgraph: 
            self.arr.append(new_input[val])
        
        arr_new_pad = pad_sequence(self.arr, batch_first=True)
        #arr_new_pack = pack_padded_sequence(arr_new_pad, self.length, batch_first=True, enforce_sorted=False)
        arr_new_pack = pack_padded_sequence(arr_new_pad, self.length, batch_first=True)
        output = self.decoder_rnn(arr_new_pack)     #LSTM

        output_unpack = pad_packed_sequence(output[0], batch_first=True) 
        output_lstm = []
        
        for v in output_unpack[0]:
            output_lstm.append(v[0])

        output_lstm = torch.stack(output_lstm) 

        c_out = torch.mean(output_unpack[0], 1)


        return obj_new_feature, node_hidden_feature, c_out 

    #  return output_lstm

    def edge_ctx(self, inp_feats, subgraph):
        self.arr = []
        for val in subgraph:
            self.arr.append(inp_feats[val])
        arr_pad = pad_sequence(self.arr, batch_first=True)
        arr_pack = pack_padded_sequence(arr_pad, self.length, batch_first=True)
        output = self.edge_ctx_rnn(arr_pack)   
        output_unpack = pad_packed_sequence(output[0], batch_first=True) 
        
        obj_new_feature = []
        for v in output_unpack[0]: 
            obj_new_feature.append(v[0])

        obj_new_feature = torch.stack(obj_new_feature, dim=0) # 
        output_edge = self.lin_edge_h(obj_new_feature)
        return output_edge

    def forward(self, x, subgraph, adj):

        obj_feas, obj_feas2, c_out = self.obj_ctx(x, subgraph) # (7600, hidden) 
        # obj_feas = self.gnn(obj_feas, adj._indices())
        # obj_feas = self.gnn(x, adj._indices())
        # print(obj_feas.shape)
        # pdb.set_trace()
        
        # obj_feas = self.lin2(obj_feas)
        # obj_feas = self.act(obj_feas)

        # obj_feas = self.obj_ctx(obj_feas, subgraph)
        # obj_feas = self.act(obj_feas)

        # obj_feas = self.gnn2(obj_feas, adj._indices())
        # obj_feas = self.lin2(obj_feas)

        

     
        # out_class = self.out_layer(obj_feas, None) # (7600, num_classes)
        # out_class = F.log_softmax(out_class, dim=1) # (7600, num_classes)
        # return out_class, obj_feas

        # return obj_feas
        
        # return self.act(obj_feas), c_out
        # return self.act(obj_feas), self.act(obj_feas2), c_out
        return self.act(obj_feas), obj_feas2, c_out
