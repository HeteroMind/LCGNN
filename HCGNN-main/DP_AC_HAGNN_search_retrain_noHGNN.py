from itertools import accumulate
from utils.data_loader import *
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from DP_AC_aggregate_frame import *
from DP_AC_Attribute_propagation_within_the_diffusion_path import *
from models.model_manager import *
from utils.data_process import *
from utils.tools import *
from scipy.sparse import dia_matrix, csr_matrix,csc_matrix, coo_matrix
import scipy.sparse as sp
from collections import deque

from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
from retrainer2 import *
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import numpy as np
from scipy.optimize import minimize
from torch.backends import cudnn
from FixedNet1 import *
import tracemalloc
from sklearn.decomposition import PCA

from DP_AC_HAGNN_noHGNN import * #这是测我自己的实验的  测HAGNN来补全   HAGNN进行下游任务

#houlaide HAGNN
import torch as th
from torch import nn
import torch.nn.functional as F
from dgl import function as fn
from dgl.nn.pytorch import edge_softmax
from dgl._ffi.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair



logger = get_logger()

SEED = 123
SEED_LIST = [123, 666, 1233, 1024, 2022]
# SEED_LIST = [123, 666, 19, 1024, 2022]
# SEED_LIST = [123, 666, 19, 42, 79]
# SEED_LIST = [123, 1233, 19, 42, 79]
# SEED_LIST = [123, 123, 123, 123, 123]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

SEARCH_LOG_PATH = 'log_output'
RETRAIN_LOG_PATH = 'retrain_log_output'

import os
import csv

def generate_file_name(dataset_name):
    return f"{dataset_name}_selected_paths.csv"

def save_paths_to_csv(paths, dataset_name, save_dir='./'):
    file_name = generate_file_name(dataset_name)
    file_path = os.path.join(save_dir, file_name)
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for path in paths:
            writer.writerow(path)
    print(f"路径已保存到文件 {file_path}")

def load_paths_from_csv(dataset_name, load_dir='./'):
    file_name = generate_file_name(dataset_name)
    file_path = os.path.join(load_dir, file_name)
    paths = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            paths.append([int(node) for node in row])
    print(f"从文件 {file_path} 加载路径完成")
    return paths
def check_paths_exist(dataset_name, load_dir='./'):
    file_name = generate_file_name(dataset_name)
    file_path = os.path.join(load_dir, file_name)
    return os.path.exists(file_path)
def get_args():
    ap = argparse.ArgumentParser(description='AutoHGNN testing for the DBLP dataset')
    ap.add_argument('--dataset', type=str, default='IMDB',
                    help='Dataset name. Default is IMDB.')  # 指定数据集名称，默认为 'IMDB'。这个参数允许用户指定要在程序中使用的数据集。
    ap.add_argument('--feats-type', type=int, default=6,  # 指定节点特征的类型，默认为 6。这个参数用于指定要在程序中使用的节点特征的类型，可以是从多种选项中选择。
                    help='Type of the node features used. ' +
                         '0 - loaded features; ' +
                         '1 - only target node features (zero vec for others); ' +
                         '2 - only target node features (id vec for others); ' +
                         '3 - all id vec. Default is 2;' +
                         '4 - only term features (id vec for others) We need to try this! Or why did we use glove!;' +
                         '5 - only term features (zero vec for others).' +
                         '6 - only valid node features (zero vec for others)'+
                         '7 - only valid node features (id vec for others)'+
                         '8 - ABLATION STURY)'
                    )
    ap.add_argument('--gnn-model', type=str, default='simpleHGN',
                    help='The gnn type in downstream task. Default is magnn、simpleHGN.')  # 指定GNN模型的类型，默认为 'gat'。这个参数用于指定在下游任务中要使用的图神经网络（GNN）模型。
    ap.add_argument('--valid-attributed-type', type=int, default=0,
                    help='The node type of valid attributed node (paper). Default is 0.')  # 指定具有属性的节点类型，默认为 0。这个参数用于指定具有属性的节点类型，通常在数据集中的某些节点具有属性，而其他节点不具有。
    ap.add_argument('--missingrate', type=float, default=0,
                    help='The node type of valid attributed node (paper). Default is 0. When --feats-type is 8, corresponding start test ',)
    # 消融实验测试不同确实率  DBLP：0.3 、0.15、0
    #                    ACM: 0.54 、0.17、 0
    #                    IMDB:0.67 、0.37、 0
    ap.add_argument('--hidden-dim', type=int, default=64,
                    help='Dimension of the node hidden state. Default is 64.')  # --hidden-dim：指定节点隐藏状态的维度，默认为 64。这个参数用于确定节点在图神经网络中的隐藏状态的维度。（）
    ap.add_argument('--num-heads', type=int, default=8,
                    help='Number of the attention heads. Default is 8.')  # 指定注意力头的数量，默认为 8。这个参数用于多头自注意力机制中的注意力头数量。
    ap.add_argument('--attn-vec-dim', type=int, default=128,
                    help='Dimension of the attention vector. Default is 128.')  # --attn-vec-dim：指定注意力向量的维度，默认为 128。这个参数用于确定注意力向量的维度，通常在自注意力机制中使用。
    ap.add_argument('--patience', type=int, default=30, help='Patience. Default is 5.')
    ap.add_argument('--batch-size', type=int, default=8,
                    help='Batch size. Default is 8.')  # --batch-size：指定训练批次大小，默认为 8。这个参数用于确定在训练中使用的批次的大小。
    ap.add_argument('--batch-size-test', type=int, default=64,
                    help='Batch size. Default is 8.')  # --batch-size-test：指定测试批次大小，默认为 32。这个参数用于确定在测试过程中使用的批次的大小。
    ap.add_argument('--repeat', type=int, default=5,
                    help='Repeat the training and testing for N times. Default is 1.')  # --repeat：指定训练和测试的重复次数，默认为 5。这个参数允许多次重复训练和测试，以获取更稳定的结果。!!!!!!!!!!!!!zhnegge训练进行五次
    ap.add_argument('--save-postfix', default='IMDB',
                    help='Postfix for the saved model and result. Default is DBLP.')  # --save-postfix：指定保存模型和结果的后缀，默认为 'DBLP'。这个参数用于为保存的模型和结果文件添加一个后缀标识。
    ap.add_argument('--feats-opt', type=str, default='1011',
                    help='0100 means 1 type nodes use our processed feature')  # --feats-opt：指定节点特征的选项，默认为 '1011'。这个参数用于控制节点特征的处理选项。
    ap.add_argument('--cuda', action='store_true', default=True,
                    help='Using GPU or not.')  # --cuda：如果存在，表示使用GPU，默认为不使用GPU。这个参数允许用户选择是否在GPU上运行程序。
    ap.add_argument('--l2norm', action='store_true', default=False,
                    help='use l2 norm in classification linear')  # --l2norm：如果存在，表示在分类线性层中使用L2范数正则化，默认为不使用。这个参数可能与正则化的类型有关。
    ap.add_argument('--time_line', type=str, default="*",
                    help='logging time')  # --time_line：指定日志时间线，默认为 '*'。这个参数用于控制日志的时间戳格式。
    ap.add_argument('--edge-feats', type=int, default=64)  # --edge-feats：指定边特征的维度，默认为 64。这个参数用于确定边特征的维度。

    ap.add_argument('--rnn-type', default='RotatE0',
                    help='Type of the aggregator. Default is RotatE0.')  # --rnn-type：指定聚合器的类型，默认为 'RotatE0'。这个参数用于确定在聚合节点嵌入时使用的聚合器类型。

    ap.add_argument('--seed', type=int, default=123,
                    help='random seed.')  # --seed：指定随机种子，默认为 123。这个参数用于确定随机性操作的随机种子。
    ap.add_argument('--use_adamw', action='store_true', default=False,
                    help='is use adamw')  # --use_adamw：如果存在，表示使用AdamW优化器，默认为不使用。这个参数可能与优化算法有关

    ap.add_argument('--neighbor_samples', type=int, default=100, help='Number of neighbors sampled. Default is 100.')

    ap.add_argument('--att_comp_dim', type=int, default=64,
                    help='Dimension of the attribute completion. Default is 64.')  # 指定属性补全的维度，默认为 64。这个参数用于确定属性补全的维度，通常在属性自动补全任务中使用。
    ap.add_argument('--use_norm', type=bool, default=False)  # --use_norm：如果存在，表示使用归一化，默认为不使用。这个参数可能与归一化操作有关。


    ap.add_argument('--schedule_step_retrain', type=int,
                    default=500)  # --schedule_step_retrain：指定重新训练中的调度步骤数量，默认为 500。这个参数用于重新训练阶段的学习率调度
    ap.add_argument('--patience_search', type=int, default=30,
                    help='Patience. Default is 8.')  # --patience_search：指定搜索过程的耐心度，默认为 8。这个参数用于确定在何时停止搜索过程，具体条件可能与训练损失或性能相关。
    ap.add_argument('--patience_retrain', type=int, default=30, help='Patience. Default is 30.')#这里可以调整为15！！！！！！！！！！！！！！

    ap.add_argument('--is_use_type_linear', type=str, default='False', help='help useTypeLinear')
    ap.add_argument('--is_use_SGD', type=str, default='False', help='help useSGD')
    ap.add_argument('--is_use_dropout', type=str, default='False', help='help useSGD')
    ap.add_argument('--momentum', type=float, default=0.9,
                    help='momentum')  # --momentum：指定动量（momentum）的值，默认为 0.9。这个参数用于调整优化算法中的动量，通常用于加速收敛。
    ap.add_argument('--inner-epoch', type=int, default=1,
                    help='Number of inner epochs. Default is 1.')  # --inner-epoch：指定内部训练周期数，默认为 1。这个参数用于指定内部训练过程中的训练周期数量。
    ap.add_argument('--use-minibatch', type=bool, default=False,
                    help='if use mini-batch')  # --use-minibatch：如果存在，表示使用小批次，默认为不使用。这个参数可能与训练过程中的批次抽样有关。

    ap.add_argument('--useSGD', action='store_true', default=False,
                    help='use SGD as supernet optimize')  # --useSGD：如果存在，表示使用随机梯度下降（SGD）作为超网络的优化器，默认为不使用。这个参数用于选择是否使用SGD优化超网络
    ap.add_argument('--useTypeLinear', action='store_true', default=False,
                    help='use each type linear')  # --useTypeLinear：如果存在，表示使用每个类型的线性层，默认为不使用。这个参数可能与神经网络结构中的线性层有关
    ap.add_argument('--usedropout', action='store_true', default=False,
                    help='use dropout')  # --usedropout：如果存在，表示使用Dropout层，默认为不使用。这个参数可能与模型的正则化有关。
    ap.add_argument('--usebn', action='store_true', default=False,
                    help='use dropout')  # --usebn：如果存在，表示使用批量归一化（Batch Normalization），默认为不使用。这个参数可能与神经网络训练中的归一化有关。
    ap.add_argument('--use_bn', action='store_true', default=False)
    ap.add_argument('--use_5seeds', action='store_true', default=True,
                    help='is use 5 different seeds')  # --use_5seeds：如果存在，表示使用5个不同的随机种子，默认为不使用。这个参数可能与随机性实验和重复运行有关。 #use_5seeds=True
    ap.add_argument('--cur_repeat', type=int, default=0,
                    help='args.cur_repeat')  # !!!!!!!!!!这里的问题！--cur_repeat：指定当前的重复次数，默认为 0。这个参数可能用于在多次运行中区分不同的重复次数。
    ap.add_argument('--no_use_fixseeds', action='store_true', default=False,
                    help='is use fixed seeds')  # --no_use_fixseeds：如果存在，表示不使用固定的随机种子，默认为不使用。这个参数可能用于随机性实验中的非固定随机性。

    ap.add_argument('--lr_rate_min', type=float, default=3e-5,
                    help='min learning rate')  # --lr_rate_min：指定最小学习率的值，默认为 3e-5。这个参数用于确定学习率衰减的最小值，以确保学习率不会过小。


    #需要调整的参数：
    ap.add_argument('--beta_1', type=float, default=0.8, help='views_loss wieght')  # 0.7
    ap.add_argument('--lr', type=float,
                    default=5e-4)  # --lr：指定学习率（learning rate）的值，默认为 5e-4。学习率是优化算法中的一个关键超参数，用于控制参数更新的步长
    ap.add_argument('--num-layers', type=int,
                    default=2)  # IMDB:6  ACM\DBLP :3--num-layers：指定神经网络中的层数，默认为 2。这个参数用于确定神经网络的深度，通常在深度学习模型中使用。

    ap.add_argument('--complete-num-layers', type=int,
                    default=2)  #
    ap.add_argument('--dropout', type=float, default=0.5)  # --dropout：指定Dropout的概率，默认为 0.5。Dropout是一种正则化技术，用于减少过拟合。
    ap.add_argument('--weight_decay', type=float,
                    default=1e-4)  # --weight_decay：指定权重衰减（weight decay）的值，默认为 1e-4。权重衰减是正则化项，用于控制参数的大小。

    ap.add_argument('--slope', type=float,
                    default=0.05)  # IMDB:0.1 ACM\DBLP :0.05--slope：指定激活函数中的斜率（slope），默认为 0.05。这个参数通常用于激活函数中的激活斜率。

    ap.add_argument('--grad_clip', type=float, default=5,
                    help='gradient clipping')  # --grad_clip：指定梯度裁剪（gradient clipping）的阈值，默认为 5。梯度裁剪用于防止梯度爆炸问题。

    ap.add_argument('--max_num_views', type=int, default=3, help='为了输入进transfromer中构建的多视角数据')

    ap.add_argument('--complete_epochs', type=int, default=1,
                    help='Number of complete_epochs. Default is 100.')  # 补全时的epoch

    ap.add_argument('--search_epoch', type=int, default=350, help='Number of epochs. Default is 50.')
    ap.add_argument('--retrain_epoch', type=int, default=500,
                    help='Number of epochs. Default is 50.')  # --retrain_epoch：指定重新训练的训练周期数，默认为 500。这个参数用于指定在重新训练过程中的训练周期数量。！！！


    ap.add_argument('--intralayers', type=int, default=2)


    args = ap.parse_args()  # 最后，通过 ap.parse_args() 解析命令行参数，并将其存储在 args 变量中，以便在程序中使用。

    return args



def set_random_seed(seed, is_cuda):
    # random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if is_cuda:
        # logger.info('Using CUDA')
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.enabled = True
        cudnn.benchmark = False
        cudnn.deterministic = True
        # cudnn.benchmark = True


def retrain(model, args, cur_repeat):
    logger = args.logger
    logger.info(f"=============== Retrain Stage Starts:=======================")
    # 创建 Retrainer 实例
    retrainer2 = Retrainer2(model.get_new_features_list(),model.get_data_info(), model.get_idx_info(), model.get_train_info(), model.get_writer(), args)

    # 循环重复次数
    #使用搜索结果创建固定模型，并进行重新训练
    fixed_model = model.create_retrain_model(model.get_new_features_list(),model.get_data_info(), model.get_idx_info(), model.get_train_info())

    fixed_model = retrainer2.retrain2(fixed_model, cur_repeat)
    #在测试集上评估模型性能
    retrainer2.test2(fixed_model, cur_repeat)

    # # '''这只是残差注意力网络'''
    # fixed_model,onehot_feature_list,node_type_feature, features_list,type_mask= model.create_retrain_model_1(model.get_data_info(),adjM,args)
    # fixed_model = retrainer1.retrain1(fixed_model,onehot_feature_list,node_type_feature, features_list,type_mask, cur_repeat)
    # # 在测试集上评估模型性能
    # retrainer1.test1(fixed_model,onehot_feature_list,node_type_feature, features_list,type_mask, cur_repeat)

    # 释放内存
    del fixed_model
    torch.cuda.empty_cache()
    gc.collect()  # 调用 gc.collect() 进行垃圾回收
    logger.info(f"############### Retrain Stage Ends! #################")
from torch import cosine_similarity
def adjust_edge_weights_by_similarity(x, edge_index, edge_attr):#edge_index =data_info[2].edges()
    """
    根据节点特征的相似度调整边权重。
    参数:
    - data: 图数据对象，需要有边索引`edge_index`和节点特征`x`。

    返回:
    - 修改权重后的图数据对象。
    """
    # 确保data对象有边权重，如果没有，则初始化为1。
    if edge_attr is None:
        edge_attr = torch.ones((edge_index[0].size(0),), dtype=torch.float)
    # 计算所有边的节点特征的相似度
    for i in range(edge_index[0].size(0)):#(edge_index[0].size(0)  emb.shape[0]
        edge_features_src = x[edge_index[0][i]]
        edge_features_dst = x[edge_index[1][i]]
        similarities = cosine_similarity(edge_features_src, edge_features_dst, dim=0)
        # 以相似度作为新的边权重
        edge_attr[i] = similarities
    return edge_attr
def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))
def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)


def multi_metapath_graph_edgevalue_premeta(hg,metapaths):
    edge_list = []
    value_list = []
    num_dic = {}
    sub = 0
    for type in hg.ntypes:
        num = hg.num_nodes(type)
        num_dic[type] = num - sub
        sub = num
    res_adj = 0
    for metapath in metapaths:
        adj = 1
        for etype in metapath:
            adj = adj * hg.adj(etype=etype, scipy_fmt='csr')
        # adj = prune(adj)
        # print("conducting",metapath)
        # print(adj)
        res_adj = res_adj + adj
    res_adj = (res_adj).tocsr()
    value_list.append(torch.tensor(res_adj.toarray(),device=hg.device,dtype=torch.float32)[res_adj.nonzero()].flatten())
    edge_list.append(res_adj.nonzero())
    value_res = torch.cat(value_list,0)
    # print(value_res)
    edge_list0 = []
    edge_list1 = []
    for item in edge_list:
        edge_list0.append(torch.tensor(item[0],device=hg.device))
    for item in edge_list:
        edge_list1.append(torch.tensor(item[1],device=hg.device))
    edge0 = torch.cat(edge_list0,0)
    edge1 = torch.cat(edge_list1,0)
    edge_res = (edge0,edge1)
    # print(edge_res)
    new_g = dgl.graph(edge_res,device=hg.device)
    # new_g = dgl.graph(edge_list[0],device=hg.device)
    new_g.edata['w'] = value_res
    # print(new_g)
    # for i in range(1,len(edge_list)):
    #     # print(edge_list[i].nonzero())
    #     new_g = dgl.add_edges(new_g,edge_list[i][0],edge_list[i][1])
    #     print(new_g)
    # print(new_g)
    # exit(0)
    return new_g,num_dic
def re_load_data(dataset, feat_type=0):
    load_fun = None
    if dataset == 'IMDB':
        load_fun = load_imdb

    elif dataset == 'ACM':
         load_fun = load_acm
    elif dataset == 'Freebase':
        feat_type = 1
        load_fun = load_freebase
    elif dataset == 'DBLP':
        load_fun = load_dblp
    # elif dataset == 'IMDB':
    #     load_fun = load_imdb
    # elif dataset == 'PubMed':
    #     load_fun = load_PubMed
    # elif dataset == 'ACM2':
    #     load_fun = load_acm2
    # elif dataset == 'DBLP2':
    #     load_fun = load_dblp2
    # elif dataset == 'DBLP4cluser':
    #     load_fun = load_dblp4cluser
    return load_fun(feat_type=feat_type)
def load_imdb(feat_type=0):
    #prefix = '../../data/IMDB'
    #dl = data_loader(prefix)

    prefix = 'IMDB'
    from utils.data_loader import data_loader
    # dl = data_loader(f'D:\pycharm_item\AUTOAC\AutoAC-main\data\IMDB')
    dl = data_loader(r'/home/yyj/MDNN-AC/AutoAC-main/data/' + prefix)
    # dl = data_loader(r'G:/yyj_code/pycharm_item/pycharm_item/HAGNN/HAGNN-main/HAGNN/data/NC/IMDB_data/' + prefix)
    link_type_dic = {0: 'md', 1: 'dm', 2: 'ma', 3: 'am', 4: 'mk', 5: 'km'}
    movie_num = dl.nodes['count'][0]
    data_dic = {}
    for link_type in dl.links['data'].keys():
        src_type = str(dl.links['meta'][link_type][0])
        dst_type = str(dl.links['meta'][link_type][1])
        data_dic[(src_type, link_type_dic[link_type], dst_type)] = dl.links['data'][link_type].nonzero() #建图只需要右边的 所以Nonzero
    hg = dgl.heterograph(data_dic)

    # author feature
    if feat_type == 0:
        '''preprocessed feature'''
        features = th.FloatTensor(dl.nodes['attr'][0])
        # print(features.shape())
    else:
        '''one-hot'''
        # indices = np.vstack((np.arange(author_num), np.arange(author_num)))
        # indices = th.LongTensor(indices)
        # values = th.FloatTensor(np.ones(author_num))
        # features = th.sparse.FloatTensor(indices, values, th.Size([author_num,author_num]))
        features = th.FloatTensor(np.eye(movie_num))

    # author labels

    labels = dl.labels_test['data'][:movie_num] + dl.labels_train['data'][:movie_num]
    labels = th.FloatTensor(labels)

    num_classes = 5

    train_valid_mask = dl.labels_train['mask'][:movie_num]
    test_mask = dl.labels_test['mask'][:movie_num]
    train_valid_indices = np.where(train_valid_mask == True)[0]
    split_index = int(0.7 * np.shape(train_valid_indices)[0])
    train_indices = train_valid_indices[:split_index]
    valid_indices = train_valid_indices[split_index:]
    train_mask = copy.copy(train_valid_mask)
    valid_mask = copy.copy(train_valid_mask)
    train_mask[valid_indices] = False
    valid_mask[train_indices] = False
    test_indices = np.where(test_mask == True)[0]

    meta_paths = [['md', 'dm'], ['ma', 'am'], ['mk', 'km']]
    return hg, features, labels, num_classes, train_indices, valid_indices, test_indices, \
           th.BoolTensor(train_mask), th.BoolTensor(valid_mask), th.BoolTensor(test_mask), meta_paths, dl
def load_dblp(feat_type=0):
    prefix = 'DBLP'
    from utils.data_loader import data_loader
    dl = data_loader(r'/home/yyj/MDNN-AC/AutoAC-main/data/' + prefix)
    # dl = data_loader(f'D:\pycharm_item\AUTOAC\AutoAC-main\data\DBLP')
    # dl = data_loader(r'G:/yyj_code/pycharm_item/pycharm_item/HAGNN/HAGNN-main/HAGNN/data/NC/DBLP_data/' + prefix)
    link_type_dic = {0: 'ap', 1: 'pc', 2: 'pt', 3: 'pa', 4: 'cp', 5: 'tp'}
    author_num = dl.nodes['count'][0]
    data_dic = {}
    # [data]本身是一个稀疏图 但是这边没用这个稀疏图 而是用了dgl
    for link_type in dl.links['data'].keys():
        src_type = str(dl.links['meta'][link_type][0])
        dst_type = str(dl.links['meta'][link_type][1])
        data_dic[(src_type, link_type_dic[link_type], dst_type)] = dl.links['data'][link_type].nonzero()
    hg = dgl.heterograph(data_dic)

    # author feature
    if feat_type == 0:
        '''preprocessed feature'''
        features = th.FloatTensor(dl.nodes['attr'][0]) # N_tar * F
    else:
        '''one-hot'''
        # indices = np.vstack((np.arange(author_num), np.arange(author_num)))
        # indices = th.LongTensor(indices)
        # values = th.FloatTensor(np.ones(author_num))
        # features = th.sparse.FloatTensor(indices, values, th.Size([author_num,author_num]))
        features = th.FloatTensor(np.eye(author_num))

    # author labels

    labels = dl.labels_test['data'][:author_num] + dl.labels_train['data'][:author_num]
    labels = [np.argmax(l) for l in labels]  # one-hot to value
    labels = th.LongTensor(labels)

    num_classes = 4

    train_valid_mask = dl.labels_train['mask'][:author_num]
    test_mask = dl.labels_test['mask'][:author_num]
    train_valid_indices = np.where(train_valid_mask == True)[0]
    split_index = int(0.7 * np.shape(train_valid_indices)[0])
    train_indices = train_valid_indices[:split_index]
    valid_indices = train_valid_indices[split_index:]
    train_mask = copy.copy(train_valid_mask)
    valid_mask = copy.copy(train_valid_mask)
    train_mask[valid_indices] = False
    valid_mask[train_indices] = False
    test_indices = np.where(test_mask == True)[0]

    meta_paths = [['ap', 'pa'], ['ap', 'pt', 'tp', 'pa'], ['ap', 'pc', 'cp', 'pa']]
    # meta_paths = [['ap', 'pt', 'tp', 'pa']]
    # meta_paths = [['ap', 'pa'], ['ap', 'pc', 'cp', 'pa']]
    return hg, features, labels, num_classes, train_indices, valid_indices, test_indices, \
           th.BoolTensor(train_mask), th.BoolTensor(valid_mask), th.BoolTensor(test_mask), meta_paths, \
           dl
def load_acm(feat_type=0):
    prefix = 'ACM'
    from utils.data_loader import data_loader
    # dl = data_loader(f'D:\pycharm_item\AUTOAC\AutoAC-main\data\ACM')
    dl = data_loader(r'/home/yyj/MDNN-AC/AutoAC-main/data/' + prefix)
    # dl = data_loader(r'G:/yyj_code/pycharm_item/pycharm_item/HAGNN/HAGNN-main/HAGNN/data/NC/ACM_data/' + prefix)
    link_type_dic = {0: 'pp', 1: '-pp', 2: 'pa', 3: 'ap', 4: 'ps', 5: 'sp', 6: 'pt', 7: 'tp'}
    paper_num = dl.nodes['count'][0]
    data_dic = {}
    for link_type in dl.links['data'].keys():
        src_type = str(dl.links['meta'][link_type][0])
        dst_type = str(dl.links['meta'][link_type][1])
        data_dic[(src_type, link_type_dic[link_type], dst_type)] = dl.links['data'][link_type].nonzero()
    hg = dgl.heterograph(data_dic)

    # paper feature
    if feat_type == 0:
        '''preprocessed feature'''
        features = th.FloatTensor(dl.nodes['attr'][0])
    else:
        '''one-hot'''
        features = th.FloatTensor(np.eye(paper_num))

    # author labels

    labels = dl.labels_test['data'][:paper_num] + dl.labels_train['data'][:paper_num]
    labels = [np.argmax(l) for l in labels]  # one-hot to value
    labels = th.LongTensor(labels)

    num_classes = 3

    train_valid_mask = dl.labels_train['mask'][:paper_num]
    test_mask = dl.labels_test['mask'][:paper_num]
    train_valid_indices = np.where(train_valid_mask == True)[0]
    split_index = int(0.7 * np.shape(train_valid_indices)[0])
    train_indices = train_valid_indices[:split_index]
    valid_indices = train_valid_indices[split_index:]
    train_mask = copy.copy(train_valid_mask)
    valid_mask = copy.copy(train_valid_mask)
    train_mask[valid_indices] = False
    valid_mask[train_indices] = False
    test_indices = np.where(test_mask == True)[0]

    meta_paths = [['pp', 'ps', 'sp'], ['-pp', 'ps', 'sp'], ['pa', 'ap'], ['ps', 'sp'], ['pt', 'tp']]
    return hg, features, labels, num_classes, train_indices, valid_indices, test_indices, \
           th.BoolTensor(train_mask), th.BoolTensor(valid_mask), th.BoolTensor(test_mask), meta_paths, \
           dl
###########################d对于Freebase数据集要吃120G显存，放弃了
def load_freebase(feat_type=1):
    prefix = 'Freebase'
    dl = data_loader(r'/home/yyj/MDNN-AC/AutoAC-main/data/' + prefix)
    link_type_dic = {0: '00', 1: '01', 2: '03', 3: '05', 4: '06',
                     5: '11',
                     6: '20', 7: '21', 8: '22', 9: '23', 10: '25',
                     11: '31', 12: '33', 13: '35',
                     14: '40', 15: '41', 16: '42', 17: '43', 18: '44', 19: '45', 20: '46', 21: '47',
                     22: '51', 23: '55',
                     24: '61', 25: '62', 26: '63', 27: '65', 28: '66', 29: '67',
                     30: '70', 31: '71', 32: '72', 33: '73', 34: '75', 35: '77',
                     36: '-00', 37: '10', 38: '30', 39: '50', 40: '60',
                     41: '-11',
                     42: '02', 43: '12', 44: '-22', 45: '32', 46: '52',
                     47: '13', 48: '-33', 49: '53',
                     50: '04', 51: '14', 52: '24', 53: '34', 54: '-44', 55: '54', 56: '64', 57: '74',
                     58: '15', 59: '-55',
                     60: '16', 61: '26', 62: '36', 63: '56', 64: '-66', 65: '76',
                     66: '07', 67: '17', 68: '27', 69: '37', 70: '57', 71: '-77',
                     }
    book_num = dl.nodes['count'][0]
    data_dic = {}
    for link_type in dl.links['data'].keys():
        src_type = str(dl.links['meta'][link_type][0])
        dst_type = str(dl.links['meta'][link_type][1])
        data_dic[(src_type, link_type_dic[link_type], dst_type)] = dl.links['data'][link_type].nonzero()
        # reverse
        if link_type_dic[link_type + 36][0] != '-':
            data_dic[(dst_type, link_type_dic[link_type + 36], src_type)] = dl.links['data'][link_type].T.nonzero()
    hg = dgl.heterograph(data_dic)

    if feat_type == 0:
        '''preprocessed feature'''
        features = th.FloatTensor(dl.nodes['attr'][0])
    else:
        '''one-hot'''
        indices = np.vstack((np.arange(book_num), np.arange(book_num)))
        indices = th.LongTensor(indices)
        values = th.FloatTensor(np.ones(book_num))
        features = th.sparse.FloatTensor(indices, values, th.Size([book_num, book_num]))
    # author labels

    labels = dl.labels_test['data'][:book_num] + dl.labels_train['data'][:book_num]
    labels = [np.argmax(l) for l in labels]  # one-hot to value
    labels = th.LongTensor(labels)

    num_classes = 7

    train_valid_mask = dl.labels_train['mask'][:book_num]
    test_mask = dl.labels_test['mask'][:book_num]
    train_valid_indices = np.where(train_valid_mask == True)[0]
    split_index = int(0.7 * np.shape(train_valid_indices)[0])
    train_indices = train_valid_indices[:split_index]
    valid_indices = train_valid_indices[split_index:]
    train_mask = copy.copy(train_valid_mask)
    valid_mask = copy.copy(train_valid_mask)
    train_mask[valid_indices] = False
    valid_mask[train_indices] = False
    test_indices = np.where(test_mask == True)[0]

    meta_paths = [['00', '00'], ['01', '10'], ['05', '52', '20'], ['04', '40'], ['04', '43', '30'], ['06', '61', '10'],
                  ['07', '70'], ]
    return hg, features, labels, num_classes, train_indices, valid_indices, test_indices, \
           th.BoolTensor(train_mask), th.BoolTensor(valid_mask), th.BoolTensor(test_mask), meta_paths,dl

    # 返回的的feature只是author的，其他的都在dl里，和源数据文件夹一致  #


def search(args):
    log_root_path = SEARCH_LOG_PATH
    log_save_file = os.path.join(log_root_path,
                                 args.dataset + '-' + args.gnn_model + '-' + 'search' + '-' + args.time_line + '.log')
    logger = get_logger(log_root_path, log_save_file)

    args.logger = logger

    logger.info(f"=============== Search Args:\n{args}")
    t = time.time()
    # load data
    features_list, adjM, type_mask, labels, train_val_test_idx, dl= load_data(args.dataset)# ,emb  # 这里由本.py文件的311行加载过来的， 然后从这一步加载进入到data.py文件,到第8行

    logger.info(f"node_type_num: {len(dl.nodes['count'])}")  # 节点类型总数
    original_feature = []
    #处理数据
    for i in range(len(features_list)):
        if isinstance(features_list[i], dia_matrix):
            # 将 dia_matrix 转换为 ndarray
            dense_feature = features_list[i].toarray()
            # 更新 features_list 中的第 i 个元素
            features_list[i] = dense_feature
        features_list[i] = torch.tensor(features_list[i], dtype=torch.float32).to(device)#.clone().detach().to(device)
        original = torch.tensor(features_list[i], dtype=torch.float32).to(device).clone().detach()
        original_feature.append(original)
    '''结合HAGNN和D2PT来进行元路径子图的局部扩散进而构造出多视角特征'''
    #HAGNN的代码
    hgorigin, _, _, _, _, _, _, _, _, _, _, _ = re_load_data(args.dataset, feat_type=0)
    if args.dataset == 'DBLP':#APCT
        # # #能用的一组
        # metapathes = [[['ap', 'pt', 'tp', 'pa'], ['ap', 'pc', 'cp', 'pa']], [['pa', 'ap'], ['pt', 'tp']]]
        #
        # select = ['0', '1']
        # select_1 = ['0', '1', '2', '3']
        # metapathes_1 = [[['ap', 'pt', 'tp', 'pa'],['ap', 'pc', 'cp', 'pa']],
        #                 [['pa', 'ap' ,'pa','ap'],['pt', 'tp','pt', 'tp']],
        #                 [['cp', 'pa', 'ap', 'pc']],
        #                 [['tp','pa', 'ap', 'pt']]]  # DBLP



        '''能用的'''
        select_1 = ['0', '1']
        metapathes = [[['ap','pt', 'tp', 'pa'],['ap', 'pc', 'cp', 'pa']],
                      [['pa',  'ap'],['pt', 'tp']],

                      ]  # DBLP
        select = ['0', '1']
        # metapathes = [[['ap', 'pt', 'tp', 'pa']],
        #               [['pa', 'ap','pt', 'tp']],
        #
        #               ]  # DBLP
        # select = ['0','1']
        metapathes_1 = [[['ap', 'pa']],
                        [['pa', 'ap']],

                       ]  # DBLP
        ''''''
        # select = ['0', '1']
        # metapathes = [[['ap', 'pt', 'tp', 'pa'], ['ap', 'pc', 'cp', 'pa']],
        #               [['pa', 'ap' ,'pa','ap'],['pt', 'tp','pt', 'tp']],
        #
        #               ]  # DBLP
        # select_1 = ['0', '1']
        # metapathes_1 = [[['ap', 'pa']],
        #                 [['pa', 'ap']],
        #
        #                 ]  # DBLP
        ''''''
        ##另加的第三个视角
        select_2 = ['0', '1']
        metapathes_2 = [[['ap', 'pt', 'tp', 'pa'], ['ap', 'pc', 'cp', 'pa']],
                        [['pt', 'tp']],

                        ]  # DBLP
        # select_2 = ['0', '1']
        # metapathes_2 = [[['ap','pt', 'tp', 'pa'],['ap', 'pc', 'cp', 'pa']],
        #                 [['pt', 'tp','pa', 'ap'],['pa', 'ap','pa', 'ap']],
        #
        #                 ]  # DBLP
        #第四
        select_3 = ['0', '1']
        metapathes_3 = [[['ap', 'pc', 'cp', 'pa']],
                        [['pa', 'ap']],

                        ]  # DBLP

    elif args.dataset == 'ACM':#  P0 A1 S2 T3
        # metapathes_1 = [[['pa', 'ap'],['ps', 'sp']],
        #     [['ap', 'pp', 'pa'],['ap', 'ps', 'sp', 'pa'], ],
        #
        #     [['tp', 'pa', 'ap', 'pt']]]#11
        # select_1 = ['0','1','3']
        # metapathes = [[['pa', 'ap'],['ps', 'sp']],
        #               [['ap', 'pt', 'tp', 'pa'],['ap', 'pa']],
        #
        #               [['tp','pp', 'pt']]]#11
        # select = ['0','1','3']

        # #第2组 在simpleHGN跑的
        # metapathes_1 = [
        #                 [['ap', 'pp','pa'], ['ap', 'ps', 'sp', 'pa']],
        #
        #                 [['tp', 'pa', 'ap', 'pt']]]  # 16
        # select_1 = ['1',  '3']
        # metapathes = [
        #               [['ap', 'pa']],
        #
        #               [['tp', 'pp', 'pt']]]#16  #有效
        #
        # select = ['1', '3']

        #第一组  0.9320113314447592   0.9326544043872943
        '''效果好的一组，且transfromer中第一层是GCN,第二层是GAT，注意力维度设置为96，就能跑'''
        # metapathes_1=[
        #             [['ap', 'pp', 'pa'],['ap', 'ps', 'sp', 'pa']],
        #
        #             [['tp', 'pa', 'ap', 'pt']]]
        # metapathes = [[['pa','ap']],[['ap', 'pa']], [['sp', 'ps']],[['tp', 'pt']]]
        # select = ['0','1','2','3']
        # select_1 = ['1','3']
        ''''''
        #第二组
        metapathes=[
                    [['ap', 'pp', 'pa'],['ap', 'ps', 'sp', 'pa']],

                    [['tp', 'pa', 'ap', 'pt']]]
        select = ['1', '3']
        metapathes_1 = [[['pa','ap']],
            [['ap',  'pa']],
            [['sp', 'ps']],
            [['tp', 'pt']]]
        select_1 = ['0','1','2' ,'3']
        ##另加的第三个视角
        select_2 = ['1', '3']
        metapathes_2 = [
                        [['ap',  'pa']],

                        [['tp', 'pa', 'ap', 'pt']]
                        ]  # ACM
        #第四
        select_3 = ['1', '3']
        metapathes_3 = [
            [['ap', 'ps', 'sp', 'pa']],

            [['tp', 'pt']]
        ]  # ACM
        # # #用原本的图结构在transfromer聚合的组别
        # #第一组：
        # metapathes_1=[ [['pa','ap']],
        #                [['ap','pp', 'pa'],['ap', 'ps', 'sp', 'pa']],
        #                [['sp','ps']],
        #                [['tp', 'pa', 'ap', 'pt']]]
        # select_1 = ['0', '1', '2', '3']
        # select = ['0', '1',  '2','3']
        # metapathes =[   [['ps', 'sp'],['pt','tp']],
        #                 [['ap', 'pa']],
        #                 [['sp','ps']],
        #                 [['tp', 'pp','pt']]]
        # #第二组
        # metapathes_1=[
        #                [['ap','pp', 'pa'],['ap', 'ps', 'sp', 'pa']],
        #
        #                [['tp', 'pa', 'ap', 'pt']]]
        # select_1 = ['1', '3']
        # select = ['1','3']
        # metapathes =[
        #               [['ap', 'pt', 'tp', 'pa'],['ap', 'ps', 'sp', 'pa']],
        #
        #               [['tp', '-pp','pt']]]

    elif args.dataset == 'IMDB':
        # #第一组！ 0.6809597664446675     0.6492345098315968
        # metapathes = [  # 为什么papap不收敛 两层 [['ma', 'am', 'ma', 'am']],
        #     [['am', 'md', 'dm', 'ma'],['am','ma']],
        #     [['km', 'ma', 'am', 'mk'],['km','mk']]]  #
        # select = ['2', '3']
        # select_1 = ['0','1','2','3']
        # metapathes_1 = [[['md', 'dm']],
        #                 [['dm', 'ma', 'am', 'md']],
        #                 [['am', 'ma']],
        #                 [['km', 'mk']]]

        # #第二组  0.6749565694431745  0.6367659470966378
        # select = ['0','1','2','3']
        # select_1 = ['0','1','2','3']
        # metapathes = [  [['md', 'dm']],
        #     [['dm','ma','am', 'md']],
        #     [['am', 'md', 'dm', 'ma']],
        #     [['km', 'ma', 'am', 'mk']]]  #
        # metapathes_1=[  [['md', 'dm']],
        #     [['dm','ma','am', 'md']],
        #     [['am', 'ma']],
        #     [['km', 'mk']]]

        # #第三组！ 0.6809597664446675     0.6492345098315968  能用的！！！！！
        # metapathes = [  # 为什么papap不收敛 两层 [['ma', 'am', 'ma', 'am']],
        #     [['am', 'md', 'dm', 'ma'],['am','ma']],
        #     [['km', 'ma', 'am', 'mk'],['km','mk']]]  #
        # select = ['2', '3']
        # select_1 = ['0','1','2','3']
        # metapathes_1 = [[['md', 'dm'],['ma', 'am'],['mk', 'km']],
        #                 [['dm', 'ma', 'am', 'md']],
        #                 [['am', 'ma']],
        #                 [['km', 'mk']]]

        metapathes = [  # 为什么papap不收敛 两层 [['ma', 'am', 'ma', 'am']],
            [['am', 'md', 'dm', 'ma']],
            [['km', 'ma', 'am', 'mk']]]  #
        select = ['2', '3']
        select_1 = ['2', '3']
        metapathes_1 = [
                        [['am', 'ma']],
                        [['km', 'mk']]]
        ##另加的第三个视角
        select_2 = ['2', '3']
        metapathes_2 = [
            [['am', 'md', 'dm', 'ma']],
            [['km', 'mk']]

        ]  # ACM
        # metapathes_2 = [
        #     [['am', 'md', 'dm', 'ma'],['am', 'ma']],
        #     [['km', 'mk'],['km', 'ma', 'am', 'mk']]
        #
        # ]  # ACM#这个也能跑，效果也还可以
        #第四
        select_3 = ['2', '3']
        metapathes_3 = [
            [['am', 'ma']],
            [['km', 'ma', 'am', 'mk']]

        ]  # ACM
    elif args.dataset == 'Freebase':
        metapathes = [ [['20','06','62'],['20','06','67','72']] ]
        select = ['2']
        metapathes_1 = [[['20','06','62']]]#['01', '10'], ['05', '52', '20'],['04','40'],['04','43','30'],['06','61','10'],['07','70']]]
        select_1 = ['2']

    #第一类元路径子图
    hgs = []
    for i, meta in enumerate(metapathes):
        hg, num_dic = multi_metapath_graph_edgevalue_premeta(hgorigin,meta)
        hgs.append(hg.to(device))


    # 第二类元路径子图
    hgs_1 = []
    for i_1, meta_1 in enumerate(metapathes_1):
        hg_1, num_dic = multi_metapath_graph_edgevalue_premeta(hgorigin, meta_1)
        hgs_1.append(hg_1.to(device))
    #额外加的第三个视角
    hgs_2 = []
    for i_2, meta_2 in enumerate(metapathes_2):
        hg_2, _ = multi_metapath_graph_edgevalue_premeta(hgorigin, meta_2)
        hgs_2.append(hg_2.to(device))

    # 额外加的第四个视角
    hgs_3 = []
    for i_3, meta_3 in enumerate(metapathes_3):
        hg_3, _ = multi_metapath_graph_edgevalue_premeta(hgorigin, meta_3)
        hgs_3.append(hg_3.to(device))


    split_list = list(num_dic.values())
    heads = [args.num_heads] * args.complete_num_layers + [1]

    node_type_split_list = [dl.nodes['count'][i] for i in range(len(dl.nodes['count']))]
    # 掩码列表的处理
    max_range_per_type = list(accumulate(node_type_split_list))
    # 根据 max_range_per_type 动态生成 ranges
    ranges = [range(0, max_range_per_type[0])]
    for i in range(1, len(max_range_per_type)):
        ranges.append(range(max_range_per_type[i - 1], max_range_per_type[i]))

    data_info, idx_info, train_info = preprocess(features_list, adjM, type_mask, labels, train_val_test_idx, dl,
                                                 args)

    '''接着开始============================================================================='''
    t1 = time.time()

    logger.info(f"=============== Prepare basic data stage finish, use {t1 - t} time.")
    return hgs,split_list,heads,select,select_1,hgs_1,\
           ranges,args,data_info, idx_info,train_info,adjM,labels,dl,type_mask,train_val_test_idx,t1, \
           select_2,hgs_2,select_3,hgs_3,original_feature





def complete(hgs,split_list,heads,select,select_1,hgs_1,ranges,args,data_info,
             idx_info,train_info,adjM,labels,dl,type_mask,train_val_test_idx,t1,select_2,hgs_2,select_3,hgs_3
             , cur_repeat,original_feature
             ):

    model = MDNNModel_5(hgs,split_list,heads,select,select_1,
                 hgs_1,
                 ranges,args,data_info, idx_info,
                 train_info,adjM,labels,dl,type_mask,train_val_test_idx,select_2,hgs_2,select_3,hgs_3
                        , cur_repeat,original_feature
                        )#,emb
    # print(model)
    model.tranin_and_val(model)
    model.test(model)
    # 释放内存
    del model
    torch.cuda.empty_cache()
    gc.collect()  # 调用 gc.collect() 进行垃圾回收
    args.logger.info(f"############### Experiments Stage Ends! ###############")
    t2 = time.time()
    args.logger.info(f"=============== one experiment stage finish, use {t2 - t1} time.")




if __name__ == '__main__':

    args = get_args()

    # if args.is_unrolled == 'True':
    #     args.unrolled = True

    if args.is_use_type_linear == 'True':
        args.useTypeLinear = True

    if args.is_use_SGD == 'True':
        args.useSGD = True

    if args.is_use_dropout == 'True':
        args.usedropout = True

    if args.dataset in ['ACM', 'IMDB']:
        args.valid_attributed_type = 0  # args.valid_attributed_type代表具有属性的节点类型。
        args.feats_opt = '0111'
    elif args.dataset == 'Freebase':
        args.feats_type = 1
        # args.valid_attributed_type = 4
        # args.feats_opt = '11110111'
        # args.valid_attributed_type = 0
        # args.feats_opt = '01111111'
        args.valid_attributed_type = 1
        args.feats_opt = '10111111'

    if args.dataset in ['DBLP', 'ACM'] and args.gnn_model == 'magnn':
        args.use_minibatch = True

    if args.gnn_model in ['gcn', 'hgt']:
        args.last_hidden_dim = args.hidden_dim
    elif args.gnn_model in ['gat', 'simpleHGN']:
        args.last_hidden_dim = args.hidden_dim * args.num_heads
    elif args.gnn_model in ['magnn']:
        if args.dataset == 'IMDB':
            args.last_hidden_dim = args.hidden_dim * args.num_heads
        elif args.dataset in ['DBLP', 'ACM']:
            args.last_hidden_dim = args.hidden_dim
        # args.last_hidden_dim = args.attn_vec_dim * args.num_heads

    if not os.path.exists('checkpoint/'):
        os.makedirs('checkpoint/')

    args.time_line = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))

    if args.use_5seeds:
        # set random seed
        hgs, split_list, heads, select,select_1,hgs_1,\
        ranges, args, data_info, idx_info,train_info, adjM, \
        labels, dl, type_mask, train_val_test_idx,t1,select_2,hgs_2,select_3,hgs_3 ,original_feature= search(args)#,emb
        for cur_repeat, seed in enumerate(SEED_LIST):
            set_random_seed(seed, args.cuda)

            args.seed = seed
            args.cur_repeat = cur_repeat

            complete(hgs,split_list,heads,select,select_1,
                 hgs_1,
                 ranges,args,data_info, idx_info,
                 train_info,adjM,labels,dl,type_mask,train_val_test_idx,t1,select_2,hgs_2,select_3,hgs_3
                             ,cur_repeat,original_feature)

    elif args.no_use_fixseeds:
        # not fix seeds
        hgs, split_list, heads, select, select_1, hgs_1, \
        ranges, args, data_info, idx_info, train_info, adjM, \
        labels, dl, type_mask, train_val_test_idx, t1, select_2, hgs_2, select_3, hgs_3, original_feature = search(
            args)  # ,emb
        for cur_repeat in range(args.repeat):
            complete(hgs, split_list, heads, select, select_1,
                     hgs_1,
                     ranges, args, data_info, idx_info,
                     train_info, adjM, labels, dl, type_mask, train_val_test_idx, t1, select_2, hgs_2, select_3, hgs_3
                     , cur_repeat, original_feature)

    else:

        set_random_seed(SEED, args.cuda)

        args.seed = SEED
        hgs, split_list, heads, select, select_1, hgs_1, \
        ranges, args, data_info, idx_info, train_info, adjM, \
        labels, dl, type_mask, train_val_test_idx, t1, select_2, hgs_2, select_3, hgs_3, original_feature = search(
            args)  # ,emb
        complete(hgs, split_list, heads, select, select_1,
                 hgs_1,
                 ranges, args, data_info, idx_info,
                 train_info, adjM, labels, dl, type_mask, train_val_test_idx, t1, select_2, hgs_2, select_3, hgs_3
                 , original_feature)  # 这


