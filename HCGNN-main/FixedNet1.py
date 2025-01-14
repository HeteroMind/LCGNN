import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import softmax
import random

from utils.tools import *
from ops.operations import *
from models import *

from torch_geometric.utils import add_self_loops, degree

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
########这是copy的，不是原代码的#############
class FixedNet1(nn.Module):
    def __init__(self, data_info, idx_info, train_info, inner_data_info, gnn_model_manager, args):
        super(FixedNet1, self).__init__()

        self.args = args
        self._logger = args.logger

        self.data_info = data_info
        self.idx_info = idx_info
        self.train_info = train_info
        self.inner_data_info = inner_data_info
        self.gnn_model_manager = gnn_model_manager
        self.gnn_model = self.gnn_model_manager.create_model_class().to(device)
        self.features_list, self.labels, self.g, self.type_mask, self.dl, self.in_dims, self.num_classes = data_info
        self.train_idx, self.val_idx, self.test_idx = idx_info
        self._criterion = train_info
        # 额外加的线性层以便于处理与下游hgnn模型相接
        # 定义额外的线性层   这两句来源于fixed_net.py文件中的  134行
        # self.hgnn_preprocess = nn.Linear(self.args.max_features_len, self.args.att_comp_dim, bias=True).to(device)
        self.hgnn_preprocess = nn.Linear(self.args.attn_vec_dim, self.args.att_comp_dim, bias=True).to(device)
        nn.init.xavier_normal_(self.hgnn_preprocess.weight, gain=1.414).to(device)
        if self.args.usebn:
            self.bn = nn.BatchNorm1d(self.args.hidden_dim)
        # self.hgnn_preprocess_1 = nn.Linear(2 * self.args.att_comp_dim, self.args.att_comp_dim, bias=True).to(device)
        # nn.init.xavier_normal_(self.hgnn_preprocess_1.weight, gain=1.414).to(device)
    #def forward(self, features_list,edge_index, mini_batch_input=None):
    def forward(self, features_list, mini_batch_input=None):
        # 额外增加线性层
        # 使用额外的线性层对特征进行处理
        #self.gnn_model.load_state_dict(torch.load('/home/yyj/MDNN-AC/AutoAC-main/checkpoint/save/net_params_{}.pt'.format(self.args.time_line)))
        features_list = self.hgnn_preprocess(features_list)
        # # Calculate average neighbor features
        # row, col = edge_index
        # deg = degree(col, features_list.size(0), dtype=features_list.dtype)
        # deg_inv = deg.pow(-1)
        # norm = deg_inv[row]
        # avg_neighbor_features = torch.zeros_like(features_list)
        # for i in range(edge_index.size(1)):
        #     avg_neighbor_features[row[i]] += norm[i] * features_list[col[i]]
        #
        # features_list = torch.cat([features_list, avg_neighbor_features], dim=1)
        #
        # # 再次使用额外的线性层对特征进行处理
        # features_list = self.hgnn_preprocess_1(features_list)
        if self.args.usebn:
            h_attributed = self.bn(features_list)
        else:
            h_attributed = features_list
        if self.args.usedropout:
            h_attributed = F.dropout(h_attributed, self.args.dropout)
            node_embedding, logits = self.gnn_model_manager.forward_pass(self.gnn_model, h_attributed, mini_batch_input)
        else:
            node_embedding, logits = self.gnn_model_manager.forward_pass(self.gnn_model, h_attributed, mini_batch_input)
        if self.args.dataset == 'IMDB':
            return node_embedding, logits, torch.sigmoid(logits)
        else:
            return node_embedding, logits, logits


