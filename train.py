import os
import argparse

import h5py
import torch
# from line_profiler import profile
from matplotlib import pyplot as plt
from numpy import mean
from sklearn.manifold import TSNE

from gcn import GCN
from graph_function import get_adj_DM, get_adj
from preprocessH5 import load_h5, preprocess_simulate_data
from utils import *
from tqdm import tqdm
from torch import optim, nn
from model import my_model
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
import scanpy as sc





def buildGraphNN(X, neighborK):
    A = kneighbors_graph(X, neighborK, mode='connectivity', metric='cosine', include_self=True)
    return A


def kld(target, pred):
    return torch.mean(torch.sum(target * torch.log(target / (pred + 1e-6)), dim=-1))

def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t()) #返回相似度矩阵

def semi_loss( z1: torch.Tensor, z2: torch.Tensor):
    tau = 10
    f = lambda x: torch.exp(x / tau)
    refl_sim = f(sim(z1, z1))
    between_sim = f(sim(z1, z2))

    return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))  #计算两个相似度矩阵之间的损失

parser = argparse.ArgumentParser()
parser.add_argument('--gnnlayers', type=int, default=3, help="Number of gnn layers")
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--dims', type=int, default=[500], help='Number of units in hidden layer 1.')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--sigma', type=float, default=0.01, help='Sigma of gaussian distribution')
parser.add_argument('--dataset', type=str, default='citeseer', help='type of dataset.')
parser.add_argument('--cluster_num', type=int, default=7, help='type of dataset.')
parser.add_argument('--device', type=str, default='cuda:1', help='device')

args = parser.parse_args()

device_ids = [ 0]
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#"Quake_10x_Spleen","Wang_Lung","Young","Adam"
for args.dataset in ["Quake_10x_Bladder"]:
    print("Using {} dataset".format(args.dataset))
    file = open("result_baseline.csv", "a+")
    print(args.dataset, file=file)
    file.close()

    if args.dataset == 'Quake_10x_Bladder':
        args.cluster_num = 4
        args.gnnlayers = 1
        args.lr = 0.01
        args.dims = [500]
        k = 30
        ng = 2000

    path = "dataset/" + args.dataset + "/data.h5"
    X, y, _, _, cl_m = load_h5(path,n_gene=ng)


    DM_adj, DM_adj_n = get_adj_DM(X, k=k)
    DM_adj = DM_adj_n
    # KNN_adj, KNN_adj_n = get_adj(X, k=k)
    KNN_adj = buildGraphNN(X, k)

    features = X
    true_labels = y
    adj1 = sp.csr_matrix(KNN_adj)
    adj2 = sp.csr_matrix(DM_adj)

    adj1 = adj1 - sp.dia_matrix((adj1.diagonal()[np.newaxis, :], [0]), shape=adj1.shape)
    adj1.eliminate_zeros()

    adj2 = adj2 - sp.dia_matrix((adj2.diagonal()[np.newaxis, :], [0]), shape=adj2.shape)
    adj2.eliminate_zeros()

    print('Laplacian Smoothing...')
    adj_norm_s1 = preprocess_graph(adj1, args.gnnlayers, norm='sym', renorm=True)
    adj_norm_s2 = preprocess_graph(adj2, args.gnnlayers, norm='sym', renorm=True)

    sm_fea_s1 = sp.csr_matrix(features).toarray()
    sm_fea_s2 = sp.csr_matrix(features).toarray()

    path1 = "dataset/{}/{}_feat_sm_{}1.npy".format(args.dataset, args.dataset, args.gnnlayers)
    path2 = "dataset/{}/{}_feat_sm_{}2.npy".format(args.dataset, args.dataset, args.gnnlayers)
    if os.path.exists(path1):
        sm_fea_s1 = sp.csr_matrix(np.load(path1, allow_pickle=True)).toarray()
    # if os.path.exists(path1):
        sm_fea_s2 = sp.csr_matrix(np.load(path2, allow_pickle=True)).toarray()
    else:
        for a1 in adj_norm_s1:
            sm_fea_s1 = a1.dot(sm_fea_s1)
        for a2 in adj_norm_s2:
            sm_fea_s2 = a2.dot(sm_fea_s2)
        np.save(path2, sm_fea_s2, allow_pickle=True)
        np.save(path1, sm_fea_s1, allow_pickle=True)

    sm_fea_s1 = torch.FloatTensor(sm_fea_s1)
    sm_fea_s2 = torch.FloatTensor(sm_fea_s2)

    adj_1st1 = (adj1 + sp.eye(adj1.shape[0])).toarray()
    adj_1st2 = (adj2 + sp.eye(adj2.shape[0])).toarray()

    acc_list = []
    nmi_list = []
    ari_list = []
    f1_list = []

    adj1 = torch.tensor(adj1.todense()).to(args.device)
    adj2 = torch.tensor(adj2.todense()).to(args.device)

    for seed in range(10):
        X_graph = []
        pre_graph = []
        setup_seed(seed)
        best_acc, best_nmi, best_ari, best_f1, prediect_labels = clustering(sm_fea_s1, true_labels, args.cluster_num)
        MLp = my_model([args.cluster_num*2] + args.dims).to(args.device)

        a =  [args.cluster_num * 2] + args.dims

        model  = GCN(nfeat=features.shape[1],
                nhid=64,
                nclass= args.cluster_num,
                dropout=0.2, device = args.device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        model = model.to(args.device)
        # model = nn.DataParallel(model, device_ids=device_ids)
        inx1 = sm_fea_s1.to(args.device)
        inx2 = sm_fea_s2.to(args.device)

        target1 = torch.FloatTensor(adj_1st1).to(args.device)
        target2 = torch.FloatTensor(adj_1st2).to(args.device)


        print('Start Training...')
        for epoch in tqdm(range(args.epochs)):
            model.train()
            # z1 , z2 = MLp(inx1, is_train=True, sigma=args.sigma)
            z1_knn = model(inx1,adj1)
            z2_knn = model(inx2,adj1)
            # z2 = z2 + torch.normal(0, torch.ones_like(z2) * args.sigma).cuda()
            l1 = semi_loss(z1_knn, z2_knn)
            l2 = semi_loss(z2_knn, z1_knn)
            ret_knn = (l1 + l2) * 0.5
            ret_knn = ret_knn.mean() if mean else ret_knn.sum()
            S_knn = z1_knn @ z2_knn.T
            S_knn = S_knn.double()

            target1 = target1.double()
            target2 = target2.double()
            # loss = F.mse_loss(S, target)
            loss_knn = F.mse_loss(S_knn, target1)

            z1_DM = model(inx1, adj2)
            z2_DM = model(inx2, adj2)
            l1 = semi_loss(z1_DM, z2_DM)
            l2 = semi_loss(z2_DM, z1_DM)
            ret_DM = (l1 + l2) * 0.5
            ret_DM = ret_DM.mean() if mean else ret_DM.sum()
            S_DM = z1_DM @ z2_DM.T
            S_DM = S_DM.double()
            loss_DM = F.mse_loss(S_DM, target2)

            z_knn = (z1_knn + z2_knn) * 0.5
            z_DM = (z1_DM + z2_DM) * 0.5

            z_em = torch.cat([z_knn,z_DM],dim = 1)
            z_em = z_em.to(torch.float32)
            z_mlp = MLp(z_em, is_train=True, sigma=args.sigma)

            loss_sim1 = F.mse_loss(S_DM, S_knn)
            loss_sim2 = F.mse_loss(S_knn, S_DM)
            loss_sim = loss_sim1 + loss_sim2

            alpna = 1    #自环
            gamma = 0.5
            beta = 0.001
            loss = (loss_knn + loss_DM) * alpna + (ret_knn  + ret_DM) * beta + loss_sim * gamma
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                model.eval()
                zz1_knn = model(inx1, adj1)
                zz2_knn = model(inx2, adj1)
                zz_knn = (zz1_knn + zz2_knn) / 2
                zz1_DM = model(inx1, adj2)
                zz2_DM = model(inx2, adj2)
                zz_DM = (zz1_DM + zz2_DM) / 2
                sigma = 0.5
                # z2 = z2 + torch.normal(0, torch.ones_like(z2) * sigma).cuda()
                hidden_emb = torch.cat([zz_knn, zz_DM], dim=1)

                hidden_emb = hidden_emb.to(torch.float32)
                z_mlp_ = MLp(hidden_emb, is_train=True, sigma=args.sigma)


                acc, nmi, ari, f1, predict_labels = clustering(z_mlp_, true_labels, args.cluster_num)
                if acc >= best_acc:
                    best_acc = acc
                    best_nmi = nmi
                    best_ari = ari
                    best_f1 = f1
                    X_graph = z_mlp_
                    pre_graph = predict_labels




        tqdm.write('acc: {}, nmi: {}, ari: {}, f1: {}'.format(best_acc, best_nmi, best_ari, best_f1))
        file = open("result_baseline.csv", "a+")
        print(best_acc, best_nmi, best_ari, best_f1, file=file)
        file.close()
        acc_list.append(best_acc)
        nmi_list.append(best_nmi)
        ari_list.append(best_ari)
        f1_list.append(best_f1)


    acc_list = np.array(acc_list)
    nmi_list = np.array(nmi_list)
    ari_list = np.array(ari_list)
    f1_list = np.array(f1_list)
    file = open("result_baseline.csv", "a+")
    print(args.gnnlayers, args.lr, args.dims, args.sigma, file=file)
    print(round(acc_list.mean(), 2), round(acc_list.std(), 2), file=file)
    print(round(nmi_list.mean(), 2), round(nmi_list.std(), 2), file=file)
    print(round(ari_list.mean(), 2), round(ari_list.std(), 2), file=file)
    print(round(f1_list.mean(), 2), round(f1_list.std(), 2), file=file)
    file.close()
    print(round(acc_list.mean(), 2))
    print(round(nmi_list.mean(), 2))
    print(round(ari_list.mean(), 2))
    print(round(f1_list.mean(), 2))
