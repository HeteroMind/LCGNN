import sys
# sys.path.append('../../')
import time
import argparse
from scipy.sparse import vstack
import torch
import torch.nn.functional as F
import numpy as np
import random
import copy
import os
import gc
from collections import Counter

import dgl
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn

from model import Network_discrete
from retrainer import Retrainer

# from utils.pytorchtools import EarlyStopping
from utils.tools import *
from utils.data import load_data
from utils.data_process import preprocess

from searcher.darts.model_search import Network_Darts
from searcher.darts.architect import Architect_Darts
from searcher.nasp.supernet import Network_Nasp
from searcher.nasp.architect import Architect_Nasp
from models.model_manager import ModelManager
from searcher import *

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

def get_args():
    ap = argparse.ArgumentParser(description='AutoHGNN testing for the DBLP dataset')
    ap.add_argument('--dataset', type=str, default='DBLP', help='Dataset name. Default is DBLP.')#指定数据集名称，默认为 'DBLP'。这个参数允许用户指定要在程序中使用的数据集。
    ap.add_argument('--feats-type', type=int, default=6,#指定节点特征的类型，默认为 6。这个参数用于指定要在程序中使用的节点特征的类型，可以是从多种选项中选择。
                    help='Type of the node features used. ' +
                        '0 - loaded features; ' +
                        '1 - only target node features (zero vec for others); ' +
                        '2 - only target node features (id vec for others); ' +
                        '3 - all id vec. Default is 2;' +
                        '4 - only term features (id vec for others) We need to try this! Or why did we use glove!;' + 
                        '5 - only term features (zero vec for others).' +
                        '6 - only valid node features (zero vec for others)')
    ap.add_argument('--gnn-model', type=str, default='simpleHGN', help='The gnn type in downstream task. Default is gat.')       #指定GNN模型的类型，默认为 'gat'。这个参数用于指定在下游任务中要使用的图神经网络（GNN）模型。
    ap.add_argument('--valid-attributed-type', type=int, default=1, help='The node type of valid attributed node (paper). Default is 1.')   #指定具有属性的节点类型，默认为 1。这个参数用于指定具有属性的节点类型，通常在数据集中的某些节点具有属性，而其他节点不具有。
    ap.add_argument('--cluster-num', type=int, default=8, help='Number of the clusters for attribute aggreation. Default is 10.')#指定用于属性聚合的簇的数量，默认为 10。这个参数用于属性聚合过程中的簇的数量。!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ap.add_argument('--cluster-eps', type=float, default=1e-5, help='epsilon for cluster end. Default is 1e-5.')#指定属性聚合结束的阈值，默认为 1e-5。这个参数用于确定何时停止属性聚合。!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    ap.add_argument('--att_comp_dim', type=int, default=64, help='Dimension of the attribute completion. Default is 64.')#指定属性补全的维度，默认为 64。这个参数用于确定属性补全的维度，通常在属性自动补全任务中使用。
    
    ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')#--hidden-dim：指定节点隐藏状态的维度，默认为 64。这个参数用于确定节点在图神经网络中的隐藏状态的维度。
    ap.add_argument('--num-heads', type=int, default=8, help='Number of the attention heads. Default is 8.')#指定注意力头的数量，默认为 8。这个参数用于多头自注意力机制中的注意力头数量。
    ap.add_argument('--attn-vec-dim', type=int, default=128, help='Dimension of the attention vector. Default is 128.')#--attn-vec-dim：指定注意力向量的维度，默认为 128。这个参数用于确定注意力向量的维度，通常在自注意力机制中使用。

    ap.add_argument('--search_epoch', type=int, default=350, help='Number of epochs. Default is 50.')#--search_epoch：指定搜索过程的训练周期数，默认为 350。这个参数用于指定在架构搜索过程中的训练周期数量。
    ap.add_argument('--retrain_epoch', type=int, default=500, help='Number of epochs. Default is 50.')#--retrain_epoch：指定重新训练的训练周期数，默认为 500。这个参数用于指定在重新训练过程中的训练周期数量。
    # ap.add_argument('--search_epoch', type=int, default=2, help='Number of epochs. Default is 50.')
    # ap.add_argument('--retrain_epoch', type=int, default=2, help='Number of epochs. Default is 50.')
    ap.add_argument('--inner-epoch', type=int, default=1, help='Number of inner epochs. Default is 1.')#--inner-epoch：指定内部训练周期数，默认为 1。这个参数用于指定内部训练过程中的训练周期数量。
    # ap.add_argument('--inner-epoch', type=int, default=20, help='Number of inner epochs. Default is 1.')

    # ap.add_argument('--patience', type=int, default=5, help='Patience. Default is 5.')
    ap.add_argument('--patience_search', type=int, default=30, help='Patience. Default is 30.')#--patience_search：指定搜索过程的耐心度，默认为 30。这个参数用于确定在何时停止搜索过程，具体条件可能与训练损失或性能相关。
    ap.add_argument('--patience_retrain', type=int, default=30, help='Patience. Default is 30.')#--patience_retrain：指定重新训练的耐心度，默认为 30。这个参数用于确定在何时停止重新训练过程，具体条件可能与训练损失或性能相关。
    
    ap.add_argument('--batch-size', type=int, default=8, help='Batch size. Default is 8.')#--batch-size：指定训练批次大小，默认为 8。这个参数用于确定在训练中使用的批次的大小。
    ap.add_argument('--batch-size-test', type=int, default=32, help='Batch size. Default is 8.')#--batch-size-test：指定测试批次大小，默认为 32。这个参数用于确定在测试过程中使用的批次的大小。

    ap.add_argument('--momentum', type=float, default=0.9, help='momentum')#--momentum：指定动量（momentum）的值，默认为 0.9。这个参数用于调整优化算法中的动量，通常用于加速收敛。
    ap.add_argument('--lr', type=float, default=5e-4)#--lr：指定学习率（learning rate）的值，默认为 5e-4。学习率是优化算法中的一个关键超参数，用于控制参数更新的步长
    # ap.add_argument('--lr', type=float, default=5e-3)
    ap.add_argument('--lr_rate_min', type=float, default=3e-5, help='min learning rate')#--lr_rate_min：指定最小学习率的值，默认为 3e-5。这个参数用于确定学习率衰减的最小值，以确保学习率不会过小。

    ap.add_argument('--num-layers', type=int, default=2)#--num-layers：指定神经网络中的层数，默认为 2。这个参数用于确定神经网络的深度，通常在深度学习模型中使用。
    ap.add_argument('--dropout', type=float, default=0.5)#--dropout：指定Dropout的概率，默认为 0.5。Dropout是一种正则化技术，用于减少过拟合。
    ap.add_argument('--weight_decay', type=float, default=1e-4)#--weight_decay：指定权重衰减（weight decay）的值，默认为 1e-4。权重衰减是正则化项，用于控制参数的大小。
    # ap.add_argument('--weight-decay', type=float, default=1e-3)3.
    ap.add_argument('--slope', type=float, default=0.05)#--slope：指定激活函数中的斜率（slope），默认为 0.05。这个参数通常用于激活函数中的激活斜率。
    ap.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')#--grad_clip：指定梯度裁剪（gradient clipping）的阈值，默认为 5。梯度裁剪用于防止梯度爆炸问题。
    
    ap.add_argument('--network-momentum', type=float, default=0.9, help='momentum')#--network-momentum：指定网络动量（network momentum）的值，默认为 0.9。这个参数与神经网络训练中的动量有关

    # ap.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')    
    # ap.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')

    ap.add_argument('--arch_learning_rate', type=float, default=5e-3, help='learning rate for arch encoding')#--arch_learning_rate：指定架构编码的学习率，默认为 5e-3。这个参数用于架构搜索任务中的学习率。!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ap.add_argument('--arch_weight_decay', type=float, default=1e-5, help='weight decay for arch encoding')#--arch_weight_decay：指定架构编码的权重衰减，默认为 1e-5。这个参数用于架构搜索任务中的权重衰减。!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    ap.add_argument('--repeat', type=int, default=5, help='Repeat the training and testing for N times. Default is 1.')#--repeat：指定训练和测试的重复次数，默认为 5。这个参数允许多次重复训练和测试，以获取更稳定的结果。!!!!!!!!!!!!!zhnegge训练进行五次
    ap.add_argument('--cluster-epoch', type=int, default=1, help='Repeat the cluster epoch each iteration. Default is 1.')#
    # ap.add_argument('--cluster-epoch', type=int, default=4, help='Repeat the cluster epoch each iteration. Default is 4.')#--cluster-epoch：指定簇聚合的周期数，默认为 4。这个参数用于确定在每个迭代中重复进行簇聚合的次数。!!!!!!!!!!!!!!!!!!!!!!!!
    '''
    对于MAGNN,--dmon_loss_alpha设为0.5(DBLP\ACM\IMBE)，M为4（DBLP\ACM）,而M为16（IMDB）
    对于SimpleHGN,--dmon_loss_alpha设为0.4(DBLP\ACM\IMBE)，M为8（DBLP）,而M为12（ACM\IMDB）
    '''
    ap.add_argument('--save-postfix', default='DBLP', help='Postfix for the saved model and result. Default is DBLP.')#--save-postfix：指定保存模型和结果的后缀，默认为 'DBLP'。这个参数用于为保存的模型和结果文件添加一个后缀标识。
    ap.add_argument('--feats-opt', type=str, default='1011', help='0100 means 1 type nodes use our processed feature')#--feats-opt：指定节点特征的选项，默认为 '1011'。这个参数用于控制节点特征的处理选项。
    ap.add_argument('--cuda', action='store_true', default=True, help='Using GPU or not.')#--cuda：如果存在，表示使用GPU，默认为不使用GPU。这个参数允许用户选择是否在GPU上运行程序。
    ap.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')#--unrolled：如果存在，表示使用一步展开的验证损失，默认为不使用。这个参数可能与神经结构搜索中的展开优化有关。
    ap.add_argument('--useSGD', action='store_true', default=False, help='use SGD as supernet optimize')#--useSGD：如果存在，表示使用随机梯度下降（SGD）作为超网络的优化器，默认为不使用。这个参数用于选择是否使用SGD优化超网络
    ap.add_argument('--useTypeLinear', action='store_true', default=False, help='use each type linear')#--useTypeLinear：如果存在，表示使用每个类型的线性层，默认为不使用。这个参数可能与神经网络结构中的线性层有关
    ap.add_argument('--l2norm', action='store_true', default=False, help='use l2 norm in classification linear')#--l2norm：如果存在，表示在分类线性层中使用L2范数正则化，默认为不使用。这个参数可能与正则化的类型有关。
    ap.add_argument('--cluster-norm', action='store_true', default=False, help='use normalization on node embedding')#--cluster-norm：如果存在，表示在节点嵌入上使用规范化（Normalization），默认为不使用。这个参数可能与节点嵌入的规范化有关。
    ap.add_argument('--usedropout', action='store_true', default=False, help='use dropout')#--usedropout：如果存在，表示使用Dropout层，默认为不使用。这个参数可能与模型的正则化有关。

    ap.add_argument('--is_unrolled', type=str, default='False', help='help unrolled')#--is_unrolled、--is_use_type_linear、--is_use_SGD、--is_use_dropout：这些参数用于传递布尔值字符串，以帮助控制程序的行为。
    ap.add_argument('--is_use_type_linear', type=str, default='False', help='help useTypeLinear')
    ap.add_argument('--is_use_SGD', type=str, default='False', help='help useSGD')
    ap.add_argument('--is_use_dropout', type=str, default='False', help='help useSGD')
    ap.add_argument('--time_line', type=str, default="*", help='logging time') #--time_line：指定日志时间线，默认为 '*'。这个参数用于控制日志的时间戳格式。

    ap.add_argument('--edge-feats', type=int, default=64)#--edge-feats：指定边特征的维度，默认为 64。这个参数用于确定边特征的维度。
    ap.add_argument('--warmup-epoch', type=int, default=0)#--warmup-epoch：指定预热周期的数量，默认为 0。预热周期通常用于模型训练的早期阶段，以确保模型参数的稳定性。
    ap.add_argument('--clusterupdate-round', type=int, default=1)#--clusterupdate-round：指定簇更新的轮数，默认为 1。这个参数用于确定进行簇更新的轮数。!!!!!!!!!!!!!!!!!!!!!!!!

    ap.add_argument('--searcher_name', type=str, default='darts')#--searcher_name：指定搜索器的名称，默认为 'darts'。这个参数可能与神经结构搜索中的搜索策略有关。!!!!!!!!!!
    
    ap.add_argument('--rnn-type', default='RotatE0', help='Type of the aggregator. Default is RotatE0.')#--rnn-type：指定聚合器的类型，默认为 'RotatE0'。这个参数用于确定在聚合节点嵌入时使用的聚合器类型。
    ap.add_argument('--neighbor-samples', type=int, default=100, help='Number of neighbors sampled. Default is 100.')#--neighbor-samples：指定采样的邻居数量，默认为 100。这个参数用于确定采样邻居节点的数量。
    
    ap.add_argument('--use-minibatch', type=bool, default=False, help='if use mini-batch')#--use-minibatch：如果存在，表示使用小批次，默认为不使用。这个参数可能与训练过程中的批次抽样有关。
    ap.add_argument('--shared_ops', action='store_true', default=True, help='ops share weights')#--shared_ops：如果存在，表示共享操作的权重，默认为不共享。这个参数可能与神经网络架构中的操作共享有关。!!!!!!!!!!!!!!!!!!!!
    ap.add_argument('--e_greedy', type=float, default=0, help='nasp e_greedy')#--e_greedy：指定NASP中的ε贪婪策略的ε值，默认为 0。这个参数可能与神经结构搜索中的探索策略有关。!!!!!!!!!!!!!!!!!!!!!!!!
    
    ap.add_argument('--usebn', action='store_true', default=False, help='use dropout')#--usebn：如果存在，表示使用批量归一化（Batch Normalization），默认为不使用。这个参数可能与神经网络训练中的归一化有关。
    
    ap.add_argument('--seed', type=int, default=123, help='random seed.')#--seed：指定随机种子，默认为 123。这个参数用于确定随机性操作的随机种子。
    
    ap.add_argument('--use_5seeds', action='store_true', default=True, help='is use 5 different seeds')#--use_5seeds：如果存在，表示使用5个不同的随机种子，默认为不使用。这个参数可能与随机性实验和重复运行有关。 #use_5seeds=True
    ap.add_argument('--no_use_fixseeds', action='store_true', default=False, help='is use fixed seeds')#--no_use_fixseeds：如果存在，表示不使用固定的随机种子，默认为不使用。这个参数可能用于随机性实验中的非固定随机性。
    
    ap.add_argument('--use_dmon', action='store_true', default=False, help='is use dmon cluster')#--use_dmon：如果存在，表示使用DMON聚类，默认为不使用。这个参数可能与聚类操作有关。!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ap.add_argument('--collapse_regularization', type=float, default=0.1, help='dmon collapse_regularization')#--collapse_regularization：指定DMON中的聚类正则化参数，默认为 0.1。这个参数用于调整DMON中的聚类正则化!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ap.add_argument('--dmon_loss_alpha', type=float, default=0.3, help='dmon collapse_regularization')#--dmon_loss_alpha：指定DMON中的损失参数，默认为 0.3。这个参数用于调整DMON中的损失函数!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    
    ap.add_argument('--tau', type=float, default=1.0, help='dmon collapse_regularization')#--tau：指定一个参数 tau，默认值为 1.0。这个参数可能用于调整某些算法中的正则化或相关参数。



    ap.add_argument('--schedule_step', type=int, default=350)#--schedule_step：指定调度步骤的数量，默认为 350。这个参数用于确定学习率调度的步骤数量
    ap.add_argument('--schedule_step_retrain', type=int, default=500)#--schedule_step_retrain：指定重新训练中的调度步骤数量，默认为 500。这个参数用于重新训练阶段的学习率调度
    ap.add_argument('--use_norm', type=bool, default=False)#--use_norm：如果存在，表示使用归一化，默认为不使用。这个参数可能与归一化操作有关。
    
    ap.add_argument('--use_adamw', action='store_true', default=False, help='is use adamw')#--use_adamw：如果存在，表示使用AdamW优化器，默认为不使用。这个参数可能与优化算法有关
    
    ap.add_argument('--use_skip', action='store_true', default=False, help='is use adamw')#--use_skip：如果存在，表示使用跳连操作，默认为不使用。这个参数可能与神经网络架构中的跳连有关。
    #ap.add_argument()
    ap.add_argument('--cur_repeat', type=int, default=0,help='args.cur_repeat')#!!!!!!!!!!这里的问题！--cur_repeat：指定当前的重复次数，默认为 0。这个参数可能用于在多次运行中区分不同的重复次数。
    #print(cur_repeat)
    args = ap.parse_args()#最后，通过 ap.parse_args() 解析命令行参数，并将其存储在 args 变量中，以便在程序中使用。
    
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

def retrain(searcher, args, cur_repeat,t1):
    # retrain_time_line = time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime(time.time()))
    # log_root_path = RETRAIN_LOG_PATH
    # log_save_file = os.path.join(log_root_path, args.dataset + '-' + args.gnn_model + '-' + 'retrain' + '-' + args.time_line + '-' + retrain_time_line + '.log')
    # logger = get_logger(log_root_path, log_save_file)
    
    # args.logger = logger
    
    logger = args.logger
    
    logger.info(f"=============== Retrain Stage Starts:")
    #search_res_file_name = searcher._save_dir_name
    search_res_file_name = searcher.discreate_file_path #这里设计带 search具体的def discreate_file_path(self):这个类，即存放位置要对应好
    dir_path = os.path.join('disrete_arch_info', search_res_file_name + '.npy')
    checkpoint = np.load(dir_path, allow_pickle=True).item()
    alpha = checkpoint['arch_params']
    node_assign = checkpoint['node_assign']
    
    logger.info(f"node_assign_Counter:\n{Counter(node_assign)}")
    
    retrainer = Retrainer(searcher._data_info, searcher._idx_info, searcher._train_info, searcher.writer, searcher._save_dir_name, args)
    
    # for cur_repeat in range(args.repeat):
    #     seed = SEED_LIST[cur_repeat]
    #     set_random_seed(seed, args.cuda)#!
        
    logger.info(f"============= repeat round: {cur_repeat}; seed: {args.seed}")
    
    fixed_model = searcher.create_retrain_model(alpha, node_assign)
    fixed_model = retrainer.retrain(fixed_model, cur_repeat)
    retrainer.test(fixed_model, cur_repeat)
    
    del fixed_model
    torch.cuda.empty_cache()
    gc.collect()

    logger.info(f"############### Retrain Stage Ends! #################")
    t2 = time.time()
    args.logger.info(f"=============== one experiment stage finish, use {t2 - t1} time.")


def search(args):
    log_root_path = SEARCH_LOG_PATH
    log_save_file = os.path.join(log_root_path, args.dataset + '-' + args.gnn_model + '-' + 'search' + '-' + args.time_line + '.log')
    logger = get_logger(log_root_path, log_save_file)
    
    args.logger = logger
    
    logger.info(f"=============== Search Args:\n{args}")
    t = time.time()
    # load data
    features_list, adjM, type_mask, labels, train_val_test_idx, dl = load_data(args.dataset)#这里由本.py文件的311行加载过来的， 然后从这一步加载进入到data.py文件,到第8行
    logger.info(f"node_type_num: {len(dl.nodes['count'])}") #节点类型总数
    # logger.info(f"type_mask ")
    
    # data process
    data_info, idx_info, train_info = preprocess(features_list, adjM, type_mask, labels, train_val_test_idx, dl, args)#这里进入data_process.py文件
    t1=time.time()
    logger.info(f"=============== Prepare basic data stage finish, use {t1 - t} time.")
    
    gnn_model_manager = ModelManager(data_info, idx_info, args)
    # gnn_model.create_model_class()

    # 调用的是NASPSearcher
    searcher = SEARCHER_NAME[args.searcher_name](data_info, idx_info, train_info, gnn_model_manager, args)#这里调用的是nasp_searcher.py中的class NASPSearcher:
    
    searcher.search()#在这一步进入nasp_searcher.py文件进行搜寻操作
    
    logger.info(f"############### Search Stage Ends! ###############")
    
    return searcher,t1

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
        args.valid_attributed_type = 0#args.valid_attributed_type代表具有属性的节点类型。
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
        args.last_hidden_dim = args.hidden_dim * args.num_heads #这行代码表明正在设置一个名为 last_hidden_dim 的参数，其值为 args.hidden_dim 乘以 args.num_heads 的结果。根据这段代码，args.hidden_dim 和 args.num_heads 可能是模型的超参数或配置参数，用于定义模型中的隐藏单元数和头数。在这种情况下，args.last_hidden_dim 似乎被设置为模型中最后一层的隐藏单元数量。一种常见的做法是在多头注意力机制中，将隐藏单元数乘以头数作为最终的隐藏单元数，以便将多头注意力的结果合并到最后一层中。这样的操作可以帮助模型更好地学习数据集中的特征关系，并且可能有助于提高模型的性能。
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
        for cur_repeat, seed in enumerate(SEED_LIST):
            
            set_random_seed(seed, args.cuda)

            args.seed = seed
            args.cur_repeat = cur_repeat
            
            searcher,t1 = search(args)#这一步从311行跳到234行然后进行数据集的加载
        
            retrain(searcher, args, cur_repeat,t1)
            
    elif args.no_use_fixseeds:
        # not fix seeds
        for cur_repeat in range(args.repeat):
            searcher = search(args)
            retrain(searcher, args, cur_repeat)
    else:
        
        set_random_seed(SEED, args.cuda)

        args.seed = SEED
        
        searcher = search(args)
        
        for cur_repeat in range(args.repeat):        
            retrain(searcher, args, cur_repeat)

