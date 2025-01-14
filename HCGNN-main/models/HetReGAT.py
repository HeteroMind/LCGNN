import torch.nn as nn
from dgl.nn.pytorch import edge_softmax, GATConv
import torch
import torch.nn.functional as F
import numpy as np
#from model.FC import FC
from scipy import sparse
import dgl
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class HGAT(nn.Module):
    def __init__(self,g,in_dims,in_dims_2,num_hidden,num_classes,num_layers,heads,activation,feat_drop,attn_drop,negative_slope,residual):
        super(HGAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.hgat_layers = nn.ModuleList()
        self.activation = activation

        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims]).to(device)

        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        self.ntfc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims_2]).to(device)
        for ntfc in self.ntfc_list:
            nn.init.xavier_normal_(ntfc.weight, gain=1.414)

        self.hgat_layers.append(GATConv(num_hidden*2, num_hidden, heads[0],feat_drop, attn_drop, negative_slope, False, self.activation)).to(device)

        for l in range(1, num_layers):
            self.hgat_layers.append(GATConv(num_hidden * heads[l-1], num_hidden, heads[l],feat_drop, attn_drop, negative_slope, residual, self.activation)).to(device)

        self.hgat_layers.append(GATConv(num_hidden * heads[-2], num_hidden, heads[-1],feat_drop, attn_drop, negative_slope, residual, None)).to(device)
        self.lines=nn.Linear(num_hidden,num_classes,bias=True).to(device)
        nn.init.xavier_normal_(self.lines.weight, gain=1.414)

    def forward(self, features_list,node_type_feature):
        h = []
        h2 = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0).to(device)
        for ntfc, feature in zip(self.ntfc_list, node_type_feature):
            h2.append(ntfc(feature.to(device)))
        h2 = torch.cat(h2, 0).to(device)
        h = torch.cat((h,h2),1).to(device)
        for l in range(self.num_layers):
            h = self.hgat_layers[l](self.g, h).flatten(1).to(device)
        h = self.hgat_layers[-1](self.g, h).mean(1).to(device)
        logits = self.lines(h).to(device)
        return logits, h


class HeReGAT_nc(nn.Module):
    def __init__(self,g,in_dim_1,in_dim_2,in_dim_3,hidden_dim,num_class,num_layer_1,num_layer_2,num_heads,f_drop,att_drop,activation,slope,res,args,dropout_rate=0.1,cuda=False,feat_opt=None):
        super(HeReGAT_nc, self).__init__()
        self.args = args
        self.feat_opt = feat_opt
        self.hidden_dim = hidden_dim
        self.fc_list = nn.ModuleList([nn.Linear(m, hidden_dim, bias=True) for m in in_dim_3])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        heads = [num_heads] * num_layer_1 + [1]
        self.layer1 = HGAT(g,in_dim_1,in_dim_2,hidden_dim,num_class,
                           num_layer_1,heads,activation,f_drop,att_drop,slope,res).to(device)

        # self.hgn_FC = FC(in_dim=hidden_dim, hidden_dim=hidden_dim, dropout=dropout_rate,
        #                        activation=F.elu, num_heads=num_heads, cuda=cuda)

        if dropout_rate > 0:
            self.feat_drop = nn.Dropout(dropout_rate).to(device)
        else:
            self.feat_drop = lambda x: x

        heads = [num_heads] * num_layer_2 + [1]
        in_dim_4 = [hidden_dim for num in range(num_class)]
        self.layer3 = HGAT(g,in_dim_4,in_dim_2,hidden_dim,num_class,
                           num_layer_2,heads, activation,f_drop,att_drop,slope,res).to(device)

    def forward(self, onehot_feature_list, node_type_feature, feat_list,type_mask):#对于inputs1，需弄清node_type_feature是什么玩意，onehot_feature_list是什么玩意,feat_list是features_list

        logits_1, emb = self.layer1(onehot_feature_list,node_type_feature)

        transformed_features = torch.zeros(type_mask.shape[0], self.hidden_dim, device=device)
        for i, fc in enumerate(self.fc_list):
            node_indices = np.where(type_mask == i)[0]
            transformed_features[node_indices] = fc(feat_list[i]).to(device)
        #feat_src = transformed_features

        # for i, opt in enumerate(self.feat_opt):
        #     if opt == 1:
        #         feat_ac = self.hgn_FC(adj[mask_list[i]][:, mask_list[node_type_src]],
        #                                emb[mask_list[i]], emb[mask_list[node_type_src]],
        #                                feat_src[mask_list[node_type_src]])
        #         transformed_features[mask_list[i]] = feat_ac
        transformed_features = self.feat_drop(transformed_features).to(device)

        node_len = []
        #transformed_feature = []
        for i in range(len(onehot_feature_list)):
            node_len.append(len(onehot_feature_list[i]))
        transformed_feature = transformed_features.split(node_len,dim=0)
        #transformed_feature.append(a)
        #a, b, c = transformed_features.split(node_len,dim=0)
        # transformed_feature.append(a)
        # transformed_feature.append(b)
        # transformed_feature.append(c)
        logits_2, h_representation = self.layer3(transformed_feature, node_type_feature)
        # if self.args.dataset == 'IMDB':
        #     logits_2=F.sigmoid(logits_2)
        # else:
        #     logits_2= logits_2
        return emb,logits_2, h_representation, transformed_features

# features_list = [torch.FloatTensor(feature) for feature in features]#！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！这一句要
# onehot_feature_list = [torch.FloatTensor(feature) for feature in features]#！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！这一句要
# in_dim_3 = [features.shape[1] for features in features_list]#in_dim_3是不同节点类型的节点的特征维度！！#！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！这一句要
# #！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！1这一段要
# node_type_feature = [[0 for c in range(1)] for r in range(len(features_list))]
# node_type_feature_init = F.one_hot(torch.arange(0, len(features_list)), num_classes=len(features_list))
# node_type_feature_init = torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
# for i in range(0, len(features_list)):
#     node_type_feature[i] = node_type_feature_init[i].expand(features_list[i].shape[0], len(node_type_feature_init)).to(
#         device).type(torch.FloatTensor)
# in_dim_2 = [features.shape[1] for features in node_type_feature]
# #！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！1这一段要
# in_dim_1 = [features.shape[0] for features in onehot_feature_list]
# for i in range(0, len(onehot_feature_list)):
#     dim = onehot_feature_list[i].shape[0]
#     indices = np.vstack((np.arange(dim), np.arange(dim)))
#     indices = torch.LongTensor(indices)
#     values = torch.FloatTensor(np.ones(dim))
#     onehot_feature_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
#
#
# adjm = sparse.csr_matrix(adjM)
# adjM = torch.FloatTensor(adjM).to(device)
# g = dgl.DGLGraph(adjm + (adjm.T))
# g = dgl.remove_self_loop(g)
# g = dgl.add_self_loop(g)
# g = g.to(device)
#
# net = HeReGAT_nc(g, in_dim_1, in_dim_2, in_dim_3, hidden_dim, num_class,
#                 num_layer_1, num_layer_2, num_heads, f_drop, att_drop, activation,
#                  slope, res, dropout_rate, False, feats_opt)