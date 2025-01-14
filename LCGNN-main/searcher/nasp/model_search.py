import copy
import numpy as np
import random
from collections import defaultdict
from sklearn import preprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# from operations import *
# from torch.autograd import Variable
# from genotypes import PRIMITIVES
# from genotypes import Genotype

from utils.tools import *
from ops.operations import *
from models import *


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

'''
这段代码定义了一个名为 MixedOp 的类，它是一个继承自 nn.Module 的PyTorch模块，用于定义神经网络中的混合操作。
以下是代码的主要部分解释：
def __init__(self, valid_type, g, in_dim, out_dim, args)：这是 MixedOp 类的构造函数。它接受以下参数：
valid_type：表示有效的节点类型。
g：表示图结构的输入。
in_dim：表示输入的特征维度。
out_dim：表示输出的特征维度。
args：包含了一些配置参数的对象。
self.g = g：将传入的图结构 g 存储为类成员变量，以便在类的其他方法中使用。

self._ops = nn.ModuleList()：创建一个空的PyTorch模块列表 _ops，用于存储不同的操作。

self._one_hot_idx = -1：初始化 _one_hot_idx 为 -1，后续代码中将用它来标识 'one-hot' 操作。

for i, primitive in enumerate(PRIMITIVES)：遍历定义的操作列表 PRIMITIVES 中的每个操作。

if primitive == 'one-hot':：检查当前操作是否为 'one-hot' 操作。

self._ops.append(None)：如果操作为 'one-hot'，则将 None 添加到 _ops 列表中，表示不执行任何操作，因为 'one-hot' 操作可能需要特殊处理，这里先占位。
op = OPS[primitive](valid_type, in_dim, out_dim, args)：根据当前循环的操作名称 primitive，从定义的 OPS 字典中选择相应的操作类，并实例化这个操作类。
这行代码会创建一个特定操作的对象 op。

self._ops.append(op)：将创建的操作对象 op 添加到 _ops 列表中，以便后续可以根据索引来选择和执行不同的操作。这里将所有操作对象依次添加到 _ops 列表中，构建了一个混合操作。'''
class MixedOp(nn.Module):
    def __init__(self, valid_type, g, in_dim, out_dim, args):#
        super(MixedOp, self).__init__()
        self.g = g
        self._ops = nn.ModuleList()
        self._one_hot_idx = -1
        for i, primitive in enumerate(PRIMITIVES):
            if primitive == 'one-hot':
                self._one_hot_idx = i
                self._ops.append(None)
                continue

            op = OPS[primitive](valid_type, in_dim, out_dim, args)
            # print(f"op shape: {op.shape}")
            self._ops.append(op)
    '''
    def forward(self, mask_matrix, x, one_hot_h=None, weights=None)：前向传播方法接受以下参数：
    mask_matrix：表示不需要梯度的掩码矩阵。
    x：表示需要梯度的输入数据。
    one_hot_h：表示一种类型的独热编码（如果操作选择 'one-hot' 的话）。
    weights：表示操作的权重。
    res = []：创建一个空列表 res，用于存储每个操作的结果。
    
    idx = 0：初始化索引 idx 为 0，用于追踪当前操作。
    
    for w, op in zip(weights, self._ops)：遍历操作权重 weights 和操作列表 _ops 中的每个操作。zip 函数用于将两个列表的元素逐一配对。
    
    if w.data > 0：检查当前操作的权重是否大于 0，以确定是否执行该操作。
    
    if idx == self._one_hot_idx:：检查当前操作是否是 'one-hot' 操作。如果是 'one-hot' 操作，执行特殊的逻辑，将操作结果添加到 res 中。
    
    else：如果当前操作不是 'one-hot'，则执行一般的操作逻辑，将操作结果添加到 res 中。
    
    return sum(res)：返回 res 中所有操作结果的和作为混合操作的最终输出。
    '''
    def forward(self, mask_matrix, x, one_hot_h=None, weights=None):
        # mask_matrix: no_grad; x: need_grad; weights: need_grad
        #TODO: this place will need use torch.select to select the correspoding indx if this one cost much
        # print(f"weights shape: {weights.shape}")
        # print(f"mask_matrix shape: {mask_matrix.shape}")
        res = []
        idx = 0
        for w, op in zip(weights, self._ops):
            if w.data > 0:
                if idx == self._one_hot_idx:
                    res.append(w * torch.spmm(mask_matrix, one_hot_h))
                else:
                    res.append(w * torch.spmm(mask_matrix, op(self.g, x)))
            else:
                res.append(w)
            idx += 1
        return sum(res)

'''
这段代码定义了一个名为 Network_Nasp 的类，它是一个继承自 nn.Module 的PyTorch模块，用于定义神经网络结构。
以下是代码的主要部分解释：
def __init__(self, g, criterion, train_val_test, type_mask, dl, in_dims, num_classes, args, e_feat=None)：这是 Network_Nasp 类的构造函数。它接受以下参数：
g：表示图结构的输入。
criterion：表示损失函数或评估标准的输入。
train_val_test：包含训练、验证和测试索引的列表，用于数据集的划分。
type_mask：表示节点类型掩码。
dl：表示数据加载器（data loader）的输入。
in_dims：表示输入特征的维度。
num_classes：表示类别的数量。
args：包含了一些配置参数的对象。
e_feat：表示额外的特征（可能是边特征）。
self.g = g：将传入的图结构 g 存储为类成员变量，以便在类的其他方法中使用。

self._criterion = criterion：将损失函数或评估标准 criterion 存储为类成员变量。

self.dl = dl：将数据加载器 dl 存储为类成员变量。

self.e_feat = e_feat：将额外的特征 e_feat 存储为类成员变量。

self.train_val_test = train_val_test：将训练、验证和测试索引的列表存储为类成员变量，以便在类的其他方法中使用。

其余的语句存储了其他输入参数和超参数，如 GNN 模型名称、输入特征维度、层数、类别数量等，也存储在类成员变量中。

这段代码的目的是初始化 Network_Nasp 类，将输入参数和配置参数存储为类成员变量，以供后续的前向传播和训练过程中使用'''
class Network_Nasp(nn.Module):
    def __init__(self, g, criterion, train_val_test, type_mask, dl, in_dims, num_classes, args, e_feat=None):
        super(Network_Nasp, self).__init__()
        # graph info
        self.g = g
        self._criterion = criterion
        self.dl = dl
        self.e_feat = e_feat

        # train val test
        self.train_val_test = train_val_test
        self.train_idx, self.val_idx, self.test_idx = train_val_test[0], train_val_test[1], train_val_test[2]

        self.gnn_model_name = args.gnn_model
        self.in_dims = in_dims
        self.num_layers = args.num_layers
        self.num_classes = num_classes
        '''
        这部分代码设置了一些关于GAT（Graph Attention Network）模型的参数以及一些与图信息相关的变量。以下是代码的主要部分解释：

        self.heads = [args.num_heads] * args.num_layers + [1]：定义了每一层的头数。args.num_heads 表示注意力头的数量，args.num_layers 表示GAT模型的层数。这里将每一层的头数都设置为 args.num_heads，最后一层只有一个头。
        
        self.dropout = args.dropout：定义了GAT模型中的丢弃率，用于随机丢弃节点特征以减小过拟合。
        
        self.slope = args.slope：定义了GAT中激活函数 LeakyReLU 的斜率（slope）参数。
        
        self.cluster_num = args.cluster_num：定义了簇的数量，可能用于聚类操作。
        
        self.valid_attr_node_type = args.valid_attributed_type：定义了有效的属性节点类型，通常与节点属性相关。
        
        self.type_mask = type_mask：表示节点类型的掩码，用于区分不同类型的节点。
        
        self.args = args：将配置参数 args 存储为类成员变量。
        
        self.all_nodes_num = dl.nodes['total']：记录图中的所有节点数量。
        
        self.all_nodes_type_num = len(dl.nodes['count'])：记录不同节点类型的数量。
        
        self.node_type_split_list = [dl.nodes['count'][i] for i in range(len(dl.nodes['count']))]：将节点类型数量存储为列表 node_type_split_list，以便在后续使用。
        
        这些参数和变量用于配置GAT模型的结构以及记录图信息，包括头数、丢弃率、斜率、节点类型信息等。这些信息将在GAT模型的构建和训练中使用'''
        # GAT params
        self.heads = [args.num_heads] * args.num_layers + [1]
        self.dropout = args.dropout
        self.slope = args.slope

        self.cluster_num = args.cluster_num
        self.valid_attr_node_type = args.valid_attributed_type
        self.type_mask = type_mask

        self.args = args

        # record graph information
        self.all_nodes_num = dl.nodes['total']
        self.all_nodes_type_num = len(dl.nodes['count'])
        # print(f"node type num: {self.all_nodes_type_num}")

        self.node_type_split_list = [dl.nodes['count'][i] for i in range(len(dl.nodes['count']))]
        '''
        self.unAttributed_nodes_num：计算了不具有属性的节点数量。使用列表推导式，遍历所有节点，检查节点是否不在指定类型的属性节点范围内，从而确定不具有属性的节点数量。

        self.unAttributed_node_id_list：创建了一个不具有属性的节点ID列表。类似地，使用列表推导式，遍历所有节点，检查节点是否不在指定类型的属性节点范围内，然后将不具有属性的节点的ID添加到列表中。
        
        random.shuffle(self.unAttributed_node_id_list)：对不具有属性的节点ID列表进行随机洗牌，以便后续的操作。
        
        self.clusternodeId2originId 和 self.originId2clusternodeId：创建了两个字典，用于将不具有属性的节点的聚类节点ID映射到原始节点ID和将原始节点ID映射到聚类节点ID。这些映射将在后续使用。
        
        self.nodeid2type：创建了一个字典，将节点ID映射到其类型。遍历所有节点类型和其范围，将节点ID映射到相应的节点类型。
        
        _init_expectation_step()：调用 _init_expectation_step 方法来初始化期望步骤。
        
        _initialize_alphas()：调用 _initialize_alphas 方法来初始化α（alpha）参数。
        
        _initialize_weights()：调用 _initialize_weights 方法来初始化权重参数。
        
        self.saved_params：创建了一个列表 self.saved_params，用于保存模型架构参数。遍历模型架构参数 _arch_parameters，将每个参数的数据克隆并添加到列表中，以备后续使用。
        
        这些操作主要用于准备与模型训练和搜索相关的数据和映射，包括不具有属性的节点的处理、节点类型映射、参数的初始化等。这些数据和映射将在后续的模型训练和搜索中使用'''
        self.unAttributed_nodes_num = sum(1 for i in range(self.all_nodes_num) if not(dl.nodes['shift'][self.valid_attr_node_type] <= i <= dl.nodes['shift_end'][self.valid_attr_node_type]))
        print(f"unAttributed nodes num: {self.unAttributed_nodes_num}")

        self.unAttributed_node_id_list = [i for i in range(self.all_nodes_num) if not(dl.nodes['shift'][self.valid_attr_node_type] <= i <= dl.nodes['shift_end'][self.valid_attr_node_type])]
        print(f"self.unAttributed_node_id_list : {self.unAttributed_node_id_list}")
        # shuffle
        # 随机打乱未标记节点的ID列表
        random.shuffle(self.unAttributed_node_id_list)
        # 创建用于映射的字典，将集群中的节点ID映射到原始节点ID以及反向映射
        self.clusternodeId2originId = {}
        self.originId2clusternodeId = {}
        for i, origin_id in enumerate(self.unAttributed_node_id_list):
            self.clusternodeId2originId[i] = origin_id
            self.originId2clusternodeId[origin_id] = i
        # 创建字典，将节点ID映射到其所属的节点类型
        self.nodeid2type = {}
        for i in range(self.all_nodes_type_num):
            for j in range(dl.nodes['shift'][i], dl.nodes['shift_end'][i] + 1):
                self.nodeid2type[j] = i
        # 初始化期望步骤
        self._init_expectation_step()
        # 初始化alpha参数
        self._initialize_alphas()
        # 初始化权重
        self._initialize_weights()
        # 保存模型参数的列表
        self.saved_params = []
        for w in self._arch_parameters:
            # 克隆参数并保存在列表中，用于后续的参数恢复
            temp = w.data.clone()
            self.saved_params.append(temp)
    
    def _init_expectation_step(self):
        # node id assign to cluster
        avg_node_num = int(self.unAttributed_nodes_num // self.cluster_num)
        remain_node_num = self.unAttributed_nodes_num % self.cluster_num
        self.init_cluster_params = {
            'each_cluster_node_num': avg_node_num,
            'last_cluster_node_num': avg_node_num + remain_node_num
        }

        temp_unAttributed_node_id_list = copy.deepcopy(self.unAttributed_node_id_list)

        # random.shuffle(temp_unAttributed_node_id_list)

        self.clusters = []
        # unAttributed node range from (0, unAttributed_nodes_num - 1)
        self.node_cluster_class = [0] * self.unAttributed_nodes_num
        # print(f"node_cluster_class len: {len(self.node_cluster_class)}")

        shift = 0
        for i in range(self.cluster_num):
            if i < self.cluster_num - 1:
                self.clusters.append(defaultdict())
                self.clusters[-1]['node_id'] = list(range(shift, shift + avg_node_num))
                # self.clusters[i]['node_id'] = list(range(shift, shift + avg_node_num))
            else:
                self.clusters.append(defaultdict())
                self.clusters[-1]['node_id'] = list(range(shift, self.unAttributed_nodes_num))
                # self.clusters[i]['node_id'] = list(range(shift, self.unAttributed_nodes_num))
            # self.clusters[i]['node_id'].sort()
            # assign the node id to its cluster-class
            for idx in self.clusters[i]['node_id']:
                # print(idx)
                self.node_cluster_class[idx] = i

            shift += avg_node_num
        
        self.node_cluster_class = np.array(self.node_cluster_class)

        # mask matrix for each cluster
        self.cluster_mask_matrix = []
        for i in range(self.cluster_num):
            cur_cluster_node_id = [(self.clusternodeId2originId[x], self.clusternodeId2originId[x], 1) for x in self.clusters[i]['node_id']]
            # self.cluster_mask_matrix.append(list_to_sp_mat(cur_cluster_node_id, (self.all_nodes_num, self.all_nodes_num)))
            self.cluster_mask_matrix.append(to_torch_sp_mat(cur_cluster_node_id, (self.all_nodes_num, self.all_nodes_num), device))

    def _initialize_alphas(self):
        num_ops = len(PRIMITIVES)
        # self.alphas = Variable(1e-3 * torch.randn(self.cluster_num, num_ops).cuda(), requires_grad=True)
        self.alphas = Variable(torch.ones(self.cluster_num, num_ops).cuda() / 2, requires_grad=True)
        self._arch_parameters = [self.alphas]
        
    def _initialize_weights(self):
        # 初始化权重函数

        # 获取输入维度和隐藏维度
        initial_dim = self.in_dims[self.valid_attr_node_type]
        hidden_dim = self.args.hidden_dim

        # 创建线性层，用于对初始维度进行预处理
        self.preprocess = nn.Linear(initial_dim, hidden_dim, bias=True)
        # self.preprocess = nn.Linear(initial_dim, hidden_dim, bias=False)
        nn.init.xavier_normal_(self.preprocess.weight, gain=1.414)

        if 'one-hot' in PRIMITIVES:
            # construct one-hot embedding weight matrix  # 构建一个独热编码嵌入权重矩阵
            self.one_hot_feature_list = []
            self.embedding_list = nn.ModuleList()
            for i in range(self.all_nodes_type_num):
                dim = self.node_type_split_list[i]
                if i == self.valid_attr_node_type:
                    self.one_hot_feature_list.append(None)
                    self.embedding_list.append(None)
                    continue
                indices = np.vstack((np.arange(dim), np.arange(dim)))
                indices = torch.LongTensor(indices)
                values = torch.FloatTensor(np.ones(dim))
                self.one_hot_feature_list.append(torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device))
                self.embedding_list.append(nn.Linear(dim, hidden_dim, bias=True))
                # self.embedding_list.append(nn.Linear(dim, hidden_dim, bias=False))
                nn.init.xavier_normal_(self.embedding_list[-1].weight, gain=1.414)

        if self.args.useTypeLinear:
            # 创建线性层列表，用于类型之间的线性映射
            self.fc_list = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=True) for i in range(self.all_nodes_type_num) if i != self.valid_attr_node_type])
            for fc in self.fc_list:
                nn.init.xavier_normal_(fc.weight, gain=1.414)
        # 初始化混合操作列表
        self._ops = nn.ModuleList()
        for k in range(self.cluster_num):
            op = MixedOp(self.valid_attr_node_type, self.g, hidden_dim, hidden_dim, self.args)
            self._ops.append(op)
        # 获取GNN模型函数
        self.gnn_model = self._get_gnn_model_func(self.gnn_model_name)
        # self.gnn_model = MODEL_NAME[self.gnn_model_name](self.g, self.in_dims, hidden_dim, self.num_classes, self.num_layers, self.heads,
        #                         F.elu, self.dropout, self.dropout, self.slope, False)

    def _get_gnn_model_func(self, model_name):
        if model_name == 'gat':
            return MODEL_NAME[self.gnn_model_name](self.g, self.in_dims, self.args.hidden_dim, self.num_classes, self.num_layers, self.heads,
                                F.elu, self.dropout, self.dropout, self.slope, False, self.args.l2norm)
        elif model_name == 'gcn':
            return MODEL_NAME[self.gnn_model_name](self.g, self.in_dims, self.args.hidden_dim, self.num_classes, self.num_layers, F.elu, self.args.dropout)
        elif model_name == 'simpleHGN':
            return MODEL_NAME[self.gnn_model_name](self.g, self.args.edge_feats, len(self.dl.links['count']) * 2 + 1, self.in_dims, self.args.hidden_dim, self.num_classes, self.num_layers, self.heads, F.elu, self.args.dropout, self.args.dropout, self.args.slope, True, 0.05)

    def arch_parameters(self):
        return self._arch_parameters
    '''这段代码定义了 _loss 方法，用于计算损失函数。以下是代码的主要部分解释：

        def _loss(self, x, y, is_valid=True)：这是 _loss 方法的定义。它接受以下参数：
        
        x：输入数据，表示节点特征。
        y：标签数据，表示节点的真实类别。
        is_valid：一个布尔值，指示是否在验证集上计算损失。默认值为 True。
        node_embedding, _, logits = self(x)：调用 self(x) 方法，将输入数据 x 传递给模型，获取节点嵌入（node_embedding）、其他信息（_，在此处未使用），以及节点的预测输出（logits）。
        
        if is_valid:：检查是否需要在验证集上计算损失。
        
        input 和 target：根据 is_valid 的值，选择合适的输入和目标数据。如果在验证集上计算损失，使用验证集的输入和目标；否则，使用训练集的输入和目标。
        
        return self._criterion(input, target)：计算损失函数，将输入数据 input 和目标数据 target 传递给损失函数 _criterion 进行计算，并返回计算得到的损失值。
        
        这段代码的主要作用是根据输入数据和目标数据计算损失值，并根据 is_valid 参数选择计算验证集损失或训练集损失。损失函数的计算通常是训练模型的核心部分，用于衡量模型的性能和指导参数更新。'''
    def _loss(self, x, y, is_valid=True):
        # if self.args.dataset == 'IMDB':
        #     node_embedding, _, logits = self(x)
        # else:
        #     node_embedding, _, logits = self(x)
        node_embedding, _, logits = self(x)
        if is_valid:
            input = logits[self.val_idx].cuda()
            target = y[self.val_idx].cuda()
        else:
            input = logits[self.train_idx].cuda()
            target = y[self.train_idx].cuda()  
        # logger.info(f"self._criterion: {self._criterion}")       
        return self._criterion(input, target)

    def execute_maximum_step(self, node_embedding):
        # node_emb = node_embedding.item()
        # 将节点嵌入转换为 numpy 数组
        node_emb = node_embedding.detach().cpu().numpy()

        # print(f"node_emb type: {type(node_emb)}; node_emb shape: {node_emb.shape}")
        # 确保节点嵌入的形状符合预期
        assert node_emb.shape[0] == self.all_nodes_num #and node_emb.shape[1] == 

        # print(f"self.originId2clusternodeId:\n{self.originId2clusternodeId}")
        # 提取未带属性的节点嵌入
        unAttributed_node_emb = []
        for i in range(self.unAttributed_nodes_num):
            # print(i)
            origin_idx = self.clusternodeId2originId[i]
            unAttributed_node_emb.append(node_emb[origin_idx].tolist())
        unAttributed_node_emb = np.array(unAttributed_node_emb)
        # 如果设置了集群归一化，则对节点嵌入进行归一化处理
        if self.args.cluster_norm:
            # scale to (0, 1)
            unAttributed_node_emb = preprocessing.scale(unAttributed_node_emb)
        # 计算新的集群中心
        new_centers = np.array([unAttributed_node_emb[self.node_cluster_class == j, :].mean(axis=0) for j in range(self.cluster_num)])

        return unAttributed_node_emb, new_centers

    def execute_expectation_step(self, unAttributed_node_emb, new_centers):
        new_assign = np.argmin(((unAttributed_node_emb[:, :, None] - new_centers.T[None, :, :]) ** 2).sum(axis=1), axis=1)
        self.node_cluster_class = copy.deepcopy(new_assign)
        self._update_cluster_info()
        return self._gen_cluster_info()
    
    def _gen_cluster_info(self):
        node_cluster_class = self.node_cluster_class
        info_str = ""
        origin_id_cluster_dict = [-1] * self.all_nodes_num
        for i in range(len(node_cluster_class)):
            original_id = self.clusternodeId2originId[i]
            origin_id_cluster_dict[original_id] = node_cluster_class[i]
            # info_str += str(original_id) + '\t' + str(node_cluster_class[i]) + ';\t'
        for i in range(self.all_nodes_num):
            info_str += str(i) + ': ' + str(origin_id_cluster_dict[i]) + ';\t'

        return info_str, origin_id_cluster_dict

    def _update_cluster_info(self):
        # empty and update
        for k in range(self.cluster_num):
            self.clusters[k]['node_id'] = []
        for i in range(self.unAttributed_nodes_num):
            self.clusters[self.node_cluster_class[i]]['node_id'].append(i)
        
        # mask matrix for each cluster
        self.cluster_mask_matrix = []
        for i in range(self.cluster_num):
            cur_cluster_node_id = [(self.clusternodeId2originId[x], self.clusternodeId2originId[x], 1) for x in self.clusters[i]['node_id']]
            # self.cluster_mask_matrix.append(list_to_sp_mat(cur_cluster_node_id, (self.all_nodes_num, self.all_nodes_num)))
            self.cluster_mask_matrix.append(to_torch_sp_mat(cur_cluster_node_id, (self.all_nodes_num, self.all_nodes_num), device))
        return self.clusters

    def save_params(self):
        for index, value in enumerate(self._arch_parameters):
            self.saved_params[index].copy_(value.data)

    def clip(self):
        clip_scale = []
        # 大于1和小于0的都置为1和0，中间的不动
        m = nn.Hardtanh(0, 1)
        for index in range(len(self._arch_parameters)):
            clip_scale.append(m(Variable(self._arch_parameters[index].data)))
        for index in range(len(self._arch_parameters)):
            self._arch_parameters[index].data = clip_scale[index].data

    def proximal_step(self, var, maxIndexs=None):
        values = var.data.cpu().numpy()
        m, n = values.shape
        alphas = []
        # 对\alpha二值化
        for i in range(m):
            for j in range(n):
                if j == maxIndexs[i]:
                    # 提前保存一下\arch_parameters里每个layer最大\alpha值，然后把这个values都做二值化
                    alphas.append(values[i][j].copy())
                    values[i][j] = 1
                else:
                    values[i][j] = 0
        
        return torch.Tensor(values).cuda()

    def binarization(self, e_greedy=0):
        self.save_params()
        for index in range(len(self._arch_parameters)):
            m,n = self._arch_parameters[index].size()
            # 随机为每个layer选一个op
            if np.random.rand() <= e_greedy:
                maxIndexs = np.random.choice(range(n), m)
            else:
                maxIndexs = self._arch_parameters[index].data.cpu().numpy().argmax(axis=1)
            self._arch_parameters[index].data = self.proximal_step(self._arch_parameters[index], maxIndexs)

    def restore(self):
        # 更新\alpha参数前做二值化，梯度下降完了之后再还原成二值化之前的
        for index in range(len(self._arch_parameters)):
            self._arch_parameters[index].data = self.saved_params[index]

    def forward(self, features_list):
        # features attribute comletion learning
        # 对输入特征进行预处理
        h_raw_attributed_transform = self.preprocess(features_list[self.valid_attr_node_type])
        # h_raw_attributed_transform = F.elu(h_raw_attributed_transform)
        # 初始化节点特征张量
        # h0 = torch.zeros(self.all_nodes_num, self.args.hidden_dim, device=device, requires_grad=True)
        h0 = torch.zeros(self.all_nodes_num, self.args.hidden_dim, device=device)
        # 获取原始属性节点的索引并将预处理后的特征赋值给对应节点
        raw_attributed_node_indices = np.where(self.type_mask == self.valid_attr_node_type)[0]
        h0[raw_attributed_node_indices] = h_raw_attributed_transform

        # logger.info(f"h0.shape: {h0.shape}\nh0:\n{h0}")
        # h = []
        # for feature in features_list:
        #     h.append(feature)
        # h = torch.cat(h, 0)
        # #TODO: zero vector meets problem? when back-propogation process
        # h0 = self.preprocess(h)
        # # h0 = F.elu(h0)
        # 初始化 one-hot 编码的节点特征
        one_hot_h = None
        if 'one-hot' in PRIMITIVES:
            # process one_hot_op
            one_hot_h = []
            for i in range(self.all_nodes_type_num):
                if i == self.valid_attr_node_type:
                    one_hot_h.append(torch.zeros((self.node_type_split_list[i], self.args.hidden_dim)).to(device))
                    continue
                dense_h = self.embedding_list[i](self.one_hot_feature_list[i])
                one_hot_h.append(dense_h)
            one_hot_h = torch.cat(one_hot_h, 0)

        # self.alphas_weight = F.softmax(self.alphas, dim=-1)

        # h_attributed = None
        # h_attributed = F.elu(h0)
        # 初始化聚合后的节点特征
        h_attributed = None
        for k in range(self.cluster_num):
            cur_k_res = self._ops[k](self.cluster_mask_matrix[k], h0, one_hot_h, self._arch_parameters[0][k])
            # logger.info(f"k: {k} cur_k_res: {cur_k_res}")
            # h_attributed = torch.add(h_attributed, cur_k_res)
            if h_attributed is None:
                h_attributed = cur_k_res
            else:
                h_attributed = torch.add(h_attributed, cur_k_res)
        # h_attributed = torch.add(h_attributed, F.elu(h0))
        h_attributed = torch.add(h_attributed, h0)

        # logger.info(f"h_attributed.shape: {h_attributed.shape}\nh_attributed\n{h_attributed}")

        # logger.info(f"self.e_feat: {self.e_feat}")
        # 根据参数设置进行节点特征的线性变换和 dropout 操作
        if self.args.useTypeLinear:
            _h = h_attributed
            _h_list = torch.split(_h, self.node_type_split_list)

            h_transform = []
            fc_idx = 0
            for i in range(self.all_nodes_type_num):
                if i == self.valid_attr_node_type:
                    h_transform.append(_h_list[i])
                    continue
                h_transform.append(self.fc_list[fc_idx](_h_list[i]))
                fc_idx += 1
            h_transform = torch.cat(h_transform, 0)

            if self.args.usedropout:
                h_transform = F.dropout(h_transform, self.args.dropout)

            # gnn part
            # 调用图神经网络模型
            node_embedding, logits = self.gnn_model(h_transform, self.e_feat)

        else:
            if self.args.usedropout:
                h_attributed = F.dropout(h_attributed, self.args.dropout)
            # 调用图神经网络模型
            node_embedding, logits = self.gnn_model(h_attributed, self.e_feat)
        # 根据数据集类型返回不同的结果
        if self.args.dataset == 'IMDB':
            return node_embedding, logits, F.sigmoid(logits)
        else:
            return node_embedding, logits, logits
    '''
    这段代码定义了一个方法 genotype，其中包含了内部函数 _parse。在 _parse 函数中，首先通过对输入的架构权重进行 argmax 操作得到架构索引，然后根据索引获取对应的基因，
    并将其拼接为字符串。在 genotype 方法中，调用了 _parse 函数，并将 softmax 操作后的架构权重传入进行处理。'''
    def genotype(self):
        def _parse(arch_weights):
            gene = []
            arch_indices = torch.argmax(arch_weights, dim=-1)
            for k in arch_indices:
                gene.append(PRIMITIVES[k])
            return '||'.join(gene)

        gene = _parse(F.softmax(self.alphas, dim=-1).data.cpu())