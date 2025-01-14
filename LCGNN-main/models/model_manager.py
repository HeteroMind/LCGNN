import torch
import dgl
import torch.nn as nn
import numpy as np

from .data_process import *
from . import *
from utils.tools import *
import torch.nn.functional as F
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class ModelManager:
    '''
    在 __init__ 方法中，初始化 ModelManager 类，接受数据信息 (data_info)、索引信息 (idx_info) 和其他参数 (args)。
    '''
    def __init__(self, data_info, idx_info, args):
        
        self.features_list, self.labels, self.g, self.type_mask, self.dl, self.in_dims, self.num_classes = data_info
        self.train_idx, self.val_idx, self.test_idx = idx_info
        
        self.args = args
        
        self.gnn_model_name = args.gnn_model
        
        self._inner_data_info = None
        self._data_process()
        # # 额外加的线性层以便于处理与下游hgnn模型相接
        # # 定义额外的线性层   这两句来源于fixed_net.py文件中的  134行
        # self.hgnn_preprocess = nn.Linear(args.max_features_len, args.hidden_dim, bias=True).to(device)
        # nn.init.xavier_normal_(self.hgnn_preprocess.weight, gain=1.414).to(device)

    '''
    在 _data_process 方法中，根据选择的图神经网络模型进行数据处理。不同的模型需要不同的数据处理。目前，主要处理了模型 simpleHGN 和 magnn。
    '''
    def _data_process(self):
        # 判断图神经网络模型是否为GAT或GCN
        if self.gnn_model_name in ['gat', 'gcn']:
            # 如果是GAT或GCN模型，暂时不进行数据处理
            # 可能的处理逻辑被注释掉，未启用
            # data_info, idx_info, train_info = process_gcn_gat(self.args)
            # self.features_list, self.labels, self.g, self.type_mask, self.dl, self.in_dims, self.num_classes = data_info
            pass
        # 判断图神经网络模型是否为simpleHGN或magnn
        elif self.gnn_model_name in ['simpleHGN', 'magnn']:
            # 如果是simpleHGN或magnn模型
            if self.gnn_model_name == 'simpleHGN':
                # 对于simpleHGN模型，调用process_simplehgnn函数进行数据处理
                # 获取处理后的边特征，存储在e_feat中
                self.e_feat = self._inner_data_info = process_simplehgnn(self.dl, self.g, self.args)
                
            elif self.gnn_model_name == 'magnn':
                # 对于magnn模型，调用process_magnn函数进行数据处理
                self._inner_data_info = data_info = process_magnn(self.args)
                # 根据数据集不同分别获取处理后的信息
                if self.args.dataset == 'IMDB':
                    self.g_lists, self.edge_metapath_indices_lists = self._inner_data_info
                elif self.args.dataset in ['DBLP', 'ACM']:
                    self.adjlists, self.edge_metapath_indices_list = self._inner_data_info
                # self.g_lists, self.edge_metapath_indices_lists = data_info
        # 判断图神经网络模型是否为hgt
        elif self.gnn_model_name == 'hgt':
            # 对于hgt模型，调用process_hgt函数进行数据处理
            self.G = process_hgt(self.dl, self.g, self.args)
    '''
    在 create_model_class 方法中，根据选择的图神经网络模型创建相应的模型对象。这里包括了 'gat', 'gcn', 'simpleHGN', 'magnn' 和 'hgt'。
    '''
    def create_model_class(self):
        # 获取当前图神经网络模型的名称
        model_name = self.gnn_model_name
        # 根据模型名称生成头部的数量列表
        self.heads = [self.args.num_heads] * self.args.num_layers + [1]
        # 根据模型名称选择不同的图神经网络模型进行实例化
        if model_name == 'gat':
            # GAT模型实例化，传入相应的参数
            model = GAT(self.g, self.in_dims, self.args.hidden_dim, self.num_classes, self.args.num_layers, self.heads,
                        F.elu, self.args.dropout, self.args.dropout, self.args.slope, False, self.args.l2norm)
        elif model_name == 'gcn':
            # GCN模型实例化，传入相应的参数
            model = GCN(self.g, self.in_dims, self.args.hidden_dim, self.num_classes, self.args.num_layers, F.elu, self.args.dropout)
        elif model_name == 'simpleHGN':
            # simpleHGN模型实例化，传入相应的参数
            model = simpleHGN(self.g, self.args.edge_feats, len(self.dl.links['count']) * 2 + 1, self.in_dims, self.args.hidden_dim, self.num_classes, self.args.num_layers, self.heads, F.elu, self.args.dropout, self.args.dropout, self.args.slope, True, 0.05)
        elif model_name == 'magnn':
            # magnn模型实例化，根据数据集不同选择不同的参数
            if self.args.dataset == 'IMDB':
                num_layers = 2
                etypes_lists = [[[0, 1], [2, 3]],
                                [[1, 0], [1, 2, 3, 0]],
                                [[3, 2], [3, 0, 1, 2]]]
                self.target_node_indices = np.where(self.type_mask == 0)[0]
                model = MAGNN_nc(num_layers, [2, 2, 2], 4, etypes_lists, self.in_dims, self.args.hidden_dim, self.num_classes, self.args.num_heads, self.args.attn_vec_dim,
                        self.args.rnn_type, self.args.dropout)
                
            elif self.args.dataset == 'DBLP':
                num_layers = 1
                etypes_list = [[0, 1], [0, 2, 3, 1], [0, 4, 5, 1]]
                # etypes_list = [[[0, 3], [0, 1, 4, 3], [0, 2, 5, 3]]]
                # self.target_node_indices = np.where(self.type_mask == 1)[0]
                # self.model = MAGNN_nc(num_layers, [3, 3, 3], 6, etypes_list, self.in_dims, self.args.hidden_dim, self.num_classes, self.args.num_heads, self.args.attn_vec_dim,
                #         self.args.rnn_type, self.args.dropout)
                model = MAGNN_nc_mb(3, 6, etypes_list, self.in_dims, self.args.hidden_dim, self.num_classes, self.args.num_heads, self.args.attn_vec_dim, self.args.rnn_type, self.args.dropout)
            elif self.args.dataset == 'ACM':
                num_layers = 1
                etypes_list = [[2, 3], [4, 5], [0, 2, 3], [0, 4, 5], [1, 2, 3], [1, 4, 5]]
                # etypes_list = [[[2, 3], [4, 5], [0, 2, 3], [0, 4, 5], [1, 2, 3], [1, 4, 5]]]
                # self.target_node_indices = np.where(self.type_mask == 0)[0]
                # self.model = MAGNN_nc(num_layers, [6], 8, etypes_list, self.in_dims, self.args.hidden_dim, self.num_classes, self.args.num_heads, self.args.attn_vec_dim,
                #         self.args.rnn_type, self.args.dropout)
                # self.model = MAGNN_nc(num_layers, [6, 6, 6, 6, 6, 6], 8, etypes_list, self.in_dims, self.args.hidden_dim, self.num_classes, self.args.num_heads, self.args.attn_vec_dim,
                #         self.args.rnn_type, self.args.dropout)
                model = MAGNN_nc_mb(6, 8, etypes_list, self.in_dims, self.args.hidden_dim, self.num_classes, self.args.num_heads, self.args.attn_vec_dim, self.args.rnn_type, self.args.dropout)


        elif model_name == 'hgt':
            # hgt模型实例化，传入相应的参数
            in_dims = [self.args.att_comp_dim for _ in range(self.dl.nodes['total'])]
            model = HGT(self.G, n_inps=in_dims, n_hid=self.args.hidden_dim, n_out=self.num_classes, n_layers=self.args.num_layers, n_heads=self.args.num_heads, use_norm = self.args.use_norm)
        # self.model.to(device)
        # 返回实例化的图神经网络模型

        #大改之后的代码！！
        elif model_name == 'HetReGat':
            features_list = [torch.cuda.FloatTensor(feature) for feature in
                             self.features_list]  # 这一句的new_data_info就是features_list
            onehot_feature_list = [torch.cuda.FloatTensor(feature) for feature in
                                   self.features_list]  # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！这一句要
            in_dim_3 = [features.shape[1] for features in
                        features_list]  # in_dim_3是不同节点类型的节点的特征维度！！#！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！这一句要
            # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！1这一段要
            node_type_feature = [[0 for c in range(1)] for r in range(len(features_list))]
            node_type_feature_init = F.one_hot(torch.arange(0, len(features_list)), num_classes=len(features_list)).to(
                device)
            # node_type_feature_init = torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]]).to(device)
            for i in range(0, len(features_list)):
                node_type_feature[i] = node_type_feature_init[i].expand(features_list[i].shape[0],
                                                                        len(node_type_feature_init)).to(device).type(
                    torch.FloatTensor)
            in_dim_2 = [features.shape[1] for features in node_type_feature]
            # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！1这一段要
            in_dim_1 = [features.shape[0] for features in onehot_feature_list]
            for i in range(0, len(onehot_feature_list)):
                dim = onehot_feature_list[i].shape[0]
                indices = np.vstack((np.arange(dim), np.arange(dim)))
                indices = torch.LongTensor(indices).to(device)
                values = torch.FloatTensor(np.ones(dim)).to(device)
                onehot_feature_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
            activation = F.elu
            # # adjm = sparse.csr_matrix(adjM)
            # # adjM = torch.cuda.FloatTensor(adjM)
            # g = dgl.DGLGraph(adjM + (adjM.T))
            # g = dgl.remove_self_loop(g)
            # g = dgl.add_self_loop(g)
            # g = g.to(device)

            model = HeReGAT_nc(self, self.g, in_dim_1, in_dim_2, in_dim_3, self.args.hidden_dim, self.num_classes,
                               self.args.num_layer_1, self.args.num_layer_2, self.args.num_heads, self.args.f_drop, self.args.att_drop,
                               activation, self.args.slope, self.args.res, self.args)




        return model
    '''
    在 forward_pass 方法中，执行前向传播。根据选择的图神经网络模型，传递输入参数，执行前向传播操作。
    '''
    def forward_pass(self, gnn_model, h, mini_batch_input=None):

        # 获取传入的图神经网络模型和模型名称
        model = gnn_model
        model_name = self.gnn_model_name
        # # 额外增加线性层
        # # 使用额外的线性层对特征进行处理
        # #
        # h = self.hgnn_preprocess(h)

        # ！！！
        # 根据模型名称执行前向传播
        if model_name == 'gat':
            return model(h)
        elif model_name == 'gcn':
            return model(h)
        elif model_name == 'simpleHGN':
            # 对于 simpleHGN 模型，传入节点特征 h 和边特征 self.e_feat 进行前向传播
            return model(h, self.e_feat)

        elif model_name == 'magnn':
            # 对于 magnn 模型，根据不同数据集选择不同的参数进行前向传播
            if self.args.dataset == 'IMDB':
                return model((self.g_lists, h, self.type_mask, self.edge_metapath_indices_lists), self.target_node_indices)
            elif self.args.dataset in ['DBLP', 'ACM']:
                # 对于 DBLP 和 ACM 数据集，从 mini_batch_input 中获取必要的参数进行前向传播
                g_list, indices_list, idx_batch_mapped_list, idx_batch = mini_batch_input
                return model((g_list, h, self.type_mask, indices_list, idx_batch_mapped_list))

            # return self.model((self.g_lists, h, self.type_mask, self.edge_metapath_indices_lists), self.target_node_indices)
        elif model_name == 'hgt':
            # 对于 hgt 模型，将节点特征 h 按照节点类型分割成 features_list，并传入模型进行前向传播
            node_type_split_list = [self.dl.nodes['count'][i] for i in range(len(self.dl.nodes['count']))]
            h_list = torch.split(h, node_type_split_list)
            features_list = [x for x in h_list]
            return model(self.G, '0', features_list)

        # 大改之后的代码！！
        elif model_name == 'HetReGat':
            features_list = [torch.cuda.FloatTensor(feature) for feature in
                             self.features_list]  # 这一句的new_data_info就是features_list
            onehot_feature_list = [torch.cuda.FloatTensor(feature) for feature in
                                   self.features_list]  # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！这一句要
            # in_dim_3 = [features.shape[1] for features in features_list]  # in_dim_3是不同节点类型的节点的特征维度！！#！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！这一句要
            # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！1这一段要
            node_type_feature = [[0 for c in range(1)] for r in range(len(features_list))]
            node_type_feature_init = F.one_hot(torch.arange(0, len(features_list)), num_classes=len(features_list)).to(
                device)
            # node_type_feature_init = torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]]).to(device)
            for i in range(0, len(features_list)):
                node_type_feature[i] = node_type_feature_init[i].expand(features_list[i].shape[0],
                                                                        len(node_type_feature_init)).to(device).type(
                    torch.FloatTensor)
            # in_dim_2 = [features.shape[1] for features in node_type_feature]
            # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！1这一段要
            # in_dim_1 = [features.shape[0] for features in onehot_feature_list]
            for i in range(0, len(onehot_feature_list)):
                dim = onehot_feature_list[i].shape[0]
                indices = np.vstack((np.arange(dim), np.arange(dim)))
                indices = torch.LongTensor(indices).to(device)
                values = torch.FloatTensor(np.ones(dim)).to(device)
                onehot_feature_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
            # activation = F.elu
            return model(onehot_feature_list, node_type_feature, self.features_list, self.type_mask)#这是用于残差注意力网络
    '''
    在 get_graph_info 方法中，获取图的信息。这个信息可能在数据处理或模型构建阶段用到。
    '''
    def get_graph_info(self):
        return self._inner_data_info
        # return self.adjlists, self.edge_metapath_indices_list
