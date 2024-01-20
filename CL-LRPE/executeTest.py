import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from models import DGI, LogReg, DGI2
from utils import process, path, args
import pdb
import random
from sklearn.metrics.pairwise import cosine_similarity
from layers.fnn import MLP


#dataset = 'cora'

dataset = 'cornell'
#dataset = 'texas' 
#dataset = 'wisconsin'
#dataset = 'film'

# training params
batch_size = 1
nb_epochs = 10000
patience = 20
lr = 0.001
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
gfeatures = features / torch.norm(features, dim=-1, keepdim=True)
cosine_similarities = torch.mm(gfeatures, gfeatures.T)

#most_dissimlar_nodes = torch.argsort(cosine_similarities, dim=1, descending=False)[:, :args.walk_length]
most_dissimlar_nodes = torch.argsort(cosine_similarities, dim=1, descending=True)[:, :args.walk_length]


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

# print(labels.shape) # (1, 2708, 7) 改完变成 (2708, 7)
# print(adj.shape) # (2708, 2708)
# print(sp_adj.shape) #稀疏矩阵 (2708, 2708)
# pdb.set_trace()

# model = DGI(hid_units, hid_units, nonlinearity, subgraph)
model = DGI(ft_size, hid_units, nonlinearity, subgraph)

optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
# optimiser = torch.optim.SGD(model.parameters(), lr = lr, momentum = 1)

if torch.cuda.is_available():
    print('Using CUDA')
    # model.cuda()
    model.to(device)
    #model = model.cuda()
    #device_ids = [0, 1]
    #model = torch.nn.DataParallel(model, device_ids=device_ids)
    # features = features.cuda()
    features = features.to(device)
    if sparse:
        # sp_adj = sp_adj.cuda()
        sp_adj = sp_adj.to(device)
    else:
        # adj = adj.cuda()
        adj = adj.to(device)
    # labels = labels.cuda()
    # idx_train = idx_train.cuda()
    # idx_val = idx_val.cuda()
    # idx_test = idx_test.cuda()
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)

b_xent = nn.BCEWithLogitsLoss() #二分类交叉熵损失函数，只能解决二分类问题 （前面会加上sigmoid函数）
xent = nn.CrossEntropyLoss() #交叉熵损失函数，用于解决多分类问题 （内部会自动加上softmax层）
cnt_wait = 0
best = 1e9
best_t = 0

# print(features.shape)
# print(adj.shape)
# pdb.set_trace()

lbl_1 = torch.ones(nb_nodes) #(1, 2708)
lbl_2 = torch.zeros(nb_nodes)
torch.cuda.empty_cache()
lbl = torch.cat((lbl_1, lbl_2)) # (1, 5416)

# lbl = lbl.cuda()
lbl = lbl.to(device)



    # print(features.shape)
    # print(shuf_fts.shape)
    # pdb.set_trace()

    # lbl_1 = torch.ones(batch_size, nb_nodes) #(1, 2708)
    # lbl_2 = torch.zeros(batch_size, nb_nodes)
    # lbl = torch.cat((lbl_1, lbl_2), 1) # (1, 5416)
    # print(idx)
    
neg = [[i] + [random.randint(0, nb_nodes - 1) for j in range(args.walk_length - 1)] for i in range(nb_nodes)] #随机构造节点游走序列
    
tmp = most_dissimlar_nodes.tolist()
    #for i in range(nb_nodes):
    #    tmp[i][0] = subgraph[0][0]
        
    #tmp = subgraph
    #for i in range(nb_nodes):
        #for j in range(27):
            #tmp[i][j+2] = random.randint(0, nb_nodes - 1)
    # tmp = [[random.randint(0, nb_nodes - 1) for j in range(args.walk_length)] for i in range(nb_nodes)] #随机构造节点游走序列
    # print(tmp[0])
    # print(len(tmp), len(tmp[0]))
    # pdb.set_trace()

    
    
    # logits = model(features, shuf_fts, sp_adj if sparse else adj, sparse, None, None, None) #(1, 5416)
 #(1, 5416)
    #loss_f = model(features, tmp, neg, sp_adj if sparse else adj, sparse, None, None, None) #(1, 5416)
    #loss_f = model(features, tmp, sp_adj if sparse else adj, sparse, None, None, None)
    #logits, loss_f = model(features, most_dissimlar_nodes, sp_adj if sparse else adj, sparse, None, None, None)
    # loss_f = model(features, tmp, sp_adj if sparse else adj, sparse, None, None, None) #(1, 5416)
    # print(logits.shape) #(5416)
    # print(lbl.shape) #(5416)
 #前2708是原图得到的特征(positive 1)，后2708是扰动特征的图得到的特征(negative 0)
    



print('Loading {}th epoch'.format(best_t))
model.load_state_dict(torch.load(dataset + '_best_dgi.pkl'))

embeds, _ = model.embed(features, tmp, sp_adj if sparse else adj, sparse, None) # (1, 2708, 512)
torch.save(embeds,'film_file')
torch.save(labels,'film_labels')

print('embeds类型',type(embeds))
#print(embeds)
print(embeds.shape) #(2708, 512)

#print(embeds[0])
#print(features[0])
#print(sum(embeds[0]))
print('sum feature:', sum(features[0]))
# pdb.set_trace()

train_embs = embeds[idx_train]
val_embs = embeds[idx_val]
test_embs = embeds[idx_test]

# print(labels.shape)
# pdb.set_trace()

# train_lbls = torch.argmax(labels[0, idx_train], dim=1)
# val_lbls = torch.argmax(labels[0, idx_val], dim=1)
# test_lbls = torch.argmax(labels[0, idx_test], dim=1)

# print(labels[idx_test])

train_lbls = torch.argmax(labels[idx_train], dim=1)
val_lbls = torch.argmax(labels[idx_val], dim=1)
test_lbls = torch.argmax(labels[idx_test], dim=1)


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

accs = []

for _ in range(5): #5次实验取平均
    #log = LogReg(hid_units, nb_classes) # linear model 512 -> 7
    log = MLP(num_layers=4, input_dim=hid_units, hidden_dim=int(hid_units/2), output_dim=nb_classes)
    opt = torch.optim.Adam(log.parameters(), lr=0.0001, weight_decay=0.0)
    # log.cuda()
    log.to(device)

    pat_steps = 0
    best_acc = torch.zeros(1)
    # best_acc = best_acc.cuda()
    best_acc = best_acc.to(device)
    for _ in range(50): #texas
    #for _ in range(9):
        log.train()
        opt.zero_grad()

        logits = log(train_embs)
        pred = torch.argmax(logits, dim=1)
        train_acc = torch.sum(pred == train_lbls).float() / train_lbls.shape[0]
        #print('train_acc:', train_acc)
        loss = xent(logits, train_lbls) 
        # print(loss)
        # pdb.set_trace()

        loss.backward()
        opt.step()

    logits = log(test_embs)
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

