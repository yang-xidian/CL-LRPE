import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F 
from models import LogReg, MOST
from utils import process, path, args
import pdb
import random
from sklearn.metrics.pairwise import cosine_similarity
from layers.fnn import MLP


#dataset = 'cora'

#dataset = 'cornell'
#dataset = 'texas'
#dataset = 'wisconsin'
dataset = 'film'

# training params
batch_size = 1
nb_epochs = 10000
patience = 20
lr = 0.0001
l2_coef = 0.0
# l2_coef = 5e-04
drop_prob = 0.0
hid_units = 100
#hid_units = 64
#hid_units = 32
sparse = True
nonlinearity = 'prelu' # special name to separate parameters

args = args.make_args()


device = torch.device('cuda:1')
#device = torch.device('cpu')

# seed = 8
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# np.random.seed(seed)

# torch.cuda.manual_seed_all(seed)
# random.seed(seed)
# torch.backends.cudnn.deterministic = True


adj, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset) #邻接矩阵，特征，标签，训练集，验证集，测试集
#features, _ = process.preprocess_features(features)

#print(adj)
#print(labels.size)
#print(idx_train)
#print(idx_val)
#print(idx_test)

# edge_index = adj._indices() # train_adj本身就是以稀疏矩阵的方式存储的，因此直接调用indices()就能得到边了

#对图中每个节点计算random walkssh
subgraph = path.get_target_random_walks(args, adj)

train_subgraph = process.getdata_from_list(idx_train, subgraph)
val_subgraph = process.getdata_from_list(idx_val, subgraph)
test_subgraph = process.getdata_from_list(idx_test, subgraph)

#subgraph = path.k_hop_neighborhood(args, adj)
print(subgraph[0])
print(type(subgraph))
#print(subgraph)

# print(idx_train)
# print(idx_test)
# print(features.shape) #(2708, 1433)
# pdb.set_trace()



nb_nodes = features.shape[0]   #节点数量
ft_size = features.shape[1]    #特征大小
# print(nb_nodes)
# print(ft_size)
#nb_classes = labels.shape[1]
nb_classes = max(labels) + 1


adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))  #对称归一化矩阵


if sparse:
    sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
else:
    adj = (adj + sp.eye(adj.shape[0])).todense()  #adj + sp.eye(adj.shape[0])表示加入自连接
# print(features[np.newaxis].shape)

#features = torch.FloatTensor(features[np.newaxis]) #np.newaxis是给features前面加上一个维度

features = torch.FloatTensor(features)
#余弦相似度取负样本
#cosine_similarities = torch.mm(features, features.T)

#most_dissimlar_nodes = torch.argsort(cosine_similarities, dim=1)[:, :args.walk_length]


if not sparse:
    adj = torch.FloatTensor(adj[np.newaxis])
    #adj = torch.FloatTensor(adj)
#labels = torch.FloatTensor(labels[np.newaxis])
#labels = torch.FloatTensor(labels)

idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)


labels = torch.LongTensor(labels)
def one_hot(x, class_count):
    return torch.eye(class_count)[x,:]
labels = one_hot(labels, nb_classes)

b_xent = nn.BCEWithLogitsLoss() #二分类交叉熵损失函数，只能解决二分类问题 （前面会加上sigmoid函数）
xent = nn.CrossEntropyLoss() #交叉熵损失函数，用于解决多分类问题 （内部会自动加上softmax层）
# print(labels.shape) # (1, 2708, 7) 改完变成 (2708, 7)
# print(adj.shape) # (2708, 2708)
# print(sp_adj.shape) #稀疏矩阵 (2708, 2708)
# pdb.set_trace()

# model = DGI(hid_units, hid_units, nonlinearity, subgraph)

#torch.save(labels,'cornell_labels')

# pdb.set_trace()

train_embs = features[idx_train]
val_embs = features[idx_val]
test_embs = features[idx_test]

# print(labels.shape)
# pdb.set_trace()

# train_lbls = torch.argmax(labels[0, idx_train], dim=1)
# val_lbls = torch.argmax(labels[0, idx_val], dim=1)
# test_lbls = torch.argmax(labels[0, idx_test], dim=1)

# print(labels[idx_test])

train_lbls = torch.argmax(labels[idx_train], dim=1)
val_lbls = torch.argmax(labels[idx_val], dim=1)
test_lbls = torch.argmax(labels[idx_test], dim=1)

train_lbls = train_lbls.to(device)
val_lbls = val_lbls.to(device)
test_lbls = test_lbls.to(device)

# print(val_lbls)
# pdb.set_trace()

# print(test_embs.shape) # (1000, 512)
# print(test_lbls.shape) # (1000, )
# print(labels[0].shape) # (2708, 7)
# print(labels[0, idx_test].shape) # (1000, 7)
# pdb.set_trace()

tot = torch.zeros(1)
# tot = tot.cuda()
tot = tot.to(device)
best = 1e9
cnt_wait = 0
accs = []
features = features.to(device)

for _ in range(5): #5次实验取平均
    #log = LogReg(hid_units, nb_classes) # linear model 512 -> 7
    model = MOST(ft_size, hid_units, nonlinearity, nb_classes)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
    #opt = torch.optim.Adam(log.parameters(), lr=0.0001, weight_decay=0.0)
    model.to(device)

    pat_steps = 0
    best_acc = torch.zeros(1)
    best_acc = best_acc.to(device)
    for epoch in range(2000): 
        model.train()
        optimiser.zero_grad()
        
        logits = model(features, train_subgraph, sp_adj if sparse else adj, sparse, None, None, None)
        loss = xent(logits, train_lbls) 
        
        
        print('epoch: {} Loss: {}'.format(epoch, loss))
        if loss < best:
            best = loss
            cnt_wait = 0
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            print('Early stopping!')
            break
        loss.backward()
        optimiser.step()

    logits = model(features, test_subgraph, sp_adj if sparse else adj, sparse, None, None, None)
    preds = torch.argmax(logits, dim=1)
    print(preds)
    print(test_lbls)
    acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
    accs.append(acc * 100)
    print(acc)
    tot += acc

print('Average accuracy:', tot / 10)

accs = torch.stack(accs)
print(accs.mean())
print(accs.std())

