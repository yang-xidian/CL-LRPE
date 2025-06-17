import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import time
from models import DGI, LogReg, DGI2
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
patience = 30
lr = 0.001
l2_coef = 0.0
# l2_coef = 5e-04
drop_prob = 0.0
#hid_units = 512
hid_units = 128
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


adj, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset)
#features, _ = process.preprocess_features(features)

#print(adj)
#print(labels.size)
#print(idx_train)
#print(idx_val)
#print(idx_test)

# edge_index = adj._indices()

#random walkssh
subgraph = path.get_target_random_walks(args, adj)
adj_1 = adj
#subgraph = path.k_hop_neighborhood(args, adj)
print(subgraph[0])
print(type(subgraph))
#print(subgraph)

# print(idx_train)
# print(idx_test)
# print(features.shape) #(2708, 1433)
# pdb.set_trace()



nb_nodes = features.shape[0]   
ft_size = features.shape[1]    
# print(nb_nodes)
# print(ft_size)
#nb_classes = labels.shape[1]
nb_classes = max(labels) + 1


adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))  


if sparse:
    sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
else:
    adj = (adj + sp.eye(adj.shape[0])).todense()  #adj + sp.eye(adj.shape[0])
# print(features[np.newaxis].shape)

#features = torch.FloatTensor(features[np.newaxis]) 

features = torch.FloatTensor(features)

gfeatures = features / torch.norm(features, dim=-1, keepdim=True)
cosine_similarities = torch.mm(gfeatures, gfeatures.T)

#most_dissimlar_nodes = torch.argsort(cosine_similarities, dim=1, descending=False)[:, :args.walk_length]
most_dissimlar_nodes = torch.argsort(cosine_similarities, dim=1, descending=True)[:, :args.feature_node_length]
tmp = most_dissimlar_nodes.tolist()

""" all_subgraph_list = []

for t in tmp:
    subgraph_list = path.generate_subgraph_sequence(args, t, adj_1)
    all_subgraph_list.append(subgraph_list) """



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

b_xent = nn.BCEWithLogitsLoss() 
xent = nn.CrossEntropyLoss()
cnt_wait = 0
best = 1e9
best_t = 0
best_val_loss = float('inf')
cnt_wait = 0 
best_epoch = 0 
# print(features.shape)
# print(adj.shape)
# pdb.set_trace()

lbl_1 = torch.ones(nb_nodes) #(1, 2708)
lbl_2 = torch.zeros(nb_nodes)
torch.cuda.empty_cache()
lbl = torch.cat((lbl_1, lbl_2)) # (1, 5416)

# lbl = lbl.cuda()
lbl = lbl.to(device)


for epoch in range(nb_epochs):
    torch.cuda.empty_cache()
# for epoch in range(2):
    start_time = time.perf_counter()
    model.train()
    optimiser.zero_grad()

    idx = np.random.permutation(nb_nodes)

    #shuf_fts = features[:, idx, :]
    shuf_fts = features[idx, :]
    # print(features.shape)
    # print(shuf_fts.shape)
    # pdb.set_trace()

    # lbl_1 = torch.ones(batch_size, nb_nodes) #(1, 2708)
    # lbl_2 = torch.zeros(batch_size, nb_nodes)
    # lbl = torch.cat((lbl_1, lbl_2), 1) # (1, 5416)
    # print(idx)
    
    neg = [[i] + [random.randint(0, nb_nodes - 1) for j in range(args.walk_length - 1)] for i in range(nb_nodes)] 
    
    tmp = most_dissimlar_nodes.tolist()
    #for i in range(nb_nodes):
    #    tmp[i][0] = subgraph[0][0]
        
    #tmp = subgraph
    #for i in range(nb_nodes):
        #for j in range(27):
            #tmp[i][j+2] = random.randint(0, nb_nodes - 1)
    # tmp = [[random.randint(0, nb_nodes - 1) for j in range(args.walk_length)] for i in range(nb_nodes)] 
    # print(tmp[0])
    # print(len(tmp), len(tmp[0]))
    # pdb.set_trace()

    if torch.cuda.is_available():
        # shuf_fts = shuf_fts.cuda()
        # lbl = lbl.cuda()
        shuf_fts = shuf_fts.to(device)
        lbl = lbl.to(device)
    
    # logits = model(features, shuf_fts, sp_adj if sparse else adj, sparse, None, None, None) #(1, 5416)
    #logits, loss_f = model(features, neg, all_subgraph_list, tmp, sp_adj if sparse else adj, sparse, None, None, None) #(1, 5416)
    logits, loss_f = model(features, neg, tmp, sp_adj if sparse else adj, sparse, None, None, None)
    #loss_f = model(features, tmp, neg, sp_adj if sparse else adj, sparse, None, None, None) #(1, 5416)
    #loss_f = model(features, tmp, sp_adj if sparse else adj, sparse, None, None, None)
    #logits, loss_f = model(features, most_dissimlar_nodes, sp_adj if sparse else adj, sparse, None, None, None)
    # loss_f = model(features, tmp, sp_adj if sparse else adj, sparse, None, None, None) #(1, 5416)
    # print(logits.shape) #(5416)
    # print(lbl.shape) #(5416)
    loss_d = b_xent(logits, lbl) 
    
    loss = loss_f
    loss = loss_f + 0.0005*loss_d
    #loss = loss_d
    #print(loss_f, loss_d)
    #loss = loss_f + 0.0001*loss_d
    # loss = loss_d
    #loss = loss_f 

    print('epoch: {} Loss: {}'.format(epoch, loss))
    # print(logits)
    # pdb.set_trace()

    model.eval()
    with torch.no_grad():
        logits_val, loss_f_val = model(features, neg, tmp, sp_adj if sparse else adj, sparse, None, None, None)
        loss_d_val = b_xent(logits_val, lbl)
        val_loss = loss_f_val + 0.0005 * loss_d_val  # Validation loss

    # Check for improvement in validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        cnt_wait = 0
        torch.save(model.state_dict(), dataset + '_best_dgi.pkl')
    else:
        cnt_wait += 1

    # If no improvement for 'patience' epochs, early stop
    if cnt_wait == patience:
        print(f"Early stopping at epoch {epoch}. Best validation loss was {best_val_loss} at epoch {best_epoch}.")
        break

    loss.backward()
    optimiser.step()

    end_time = time.perf_counter()
    print('runtime：', end_time-start_time)


print('Loading {}th epoch'.format(best_t))
model.load_state_dict(torch.load(dataset + '_best_dgi.pkl'))

embeds, _ = model.embed(features, tmp, sp_adj if sparse else adj, sparse, None) # (1, 2708, 512)
#torch.save(embeds,'film_file')
#torch.save(labels,'film_labels')

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


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
def get_all_cv_score(emb, G, G_label, clf):
    ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    tot = 0
    for i in clf:  # svc_linear,svc_rbf,
        k = []
        k1 = []
        print(i)
        for test_size in tqdm_notebook(ratios):
            train, test, train_label, test_label = train_test_split(
                emb,
                G_label,
                test_size=1 - test_size)
            try:
#                 print('try:',train.shape)
                scores_clf = cross_validate(i,
                                            train,
                                            train_label,
                                            cv=5,
                                            scoring=['f1_micro', 'f1_macro'],
                                            n_jobs=10,
                                            verbose=0)
            except:
#                 print('except:',train.shape)
                scores_clf = cross_validate(i,
                                            train,
                                            train_label,
                                            cv=5,
                                            scoring=['f1_micro', 'f1_macro'],
                                            n_jobs=10,
                                            verbose=0)
            k.append([scores_clf['test_f1_micro'].mean(),
                    scores_clf['test_f1_micro'].std() * 2,
                    scores_clf['test_f1_macro'].mean(),
                    scores_clf['test_f1_macro'].std() * 2])
    return k


node_embedding = embeds.cpu().detach().numpy()
node_embedding = pd.DataFrame(node_embedding)
lab = np.argmax(labels.cpu().detach().numpy(), axis=1)
k = get_all_cv_score(node_embedding, G, lab, [LogisticRegression(n_jobs=10)])
tr = pd.DataFrame(k).T
ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
tr.columns = ['ratio {}'.format(j) for j in ratios]
tr.index = ['train-micro', 'micro-std','train-macro','macro-std']
print(tr)

