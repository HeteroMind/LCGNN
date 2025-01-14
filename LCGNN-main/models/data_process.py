import torch
import torch.nn as nn
import dgl
import networkx as nx
import numpy as np
from utils import *
# import sys
# print(sys.path)

'''
这段代码似乎是用于处理 Heterogeneous Graph Transformer（HGT）中的数据预处理工作。它包含了以下主要步骤：
构建边字典 edge_dict：

遍历了输入的链接信息（在 dl.links['meta'] 中），并将元路径映射到边信息的字典 edge_dict 中。
每个元路径都被转换为一个字典键，对应的值是一个元组，其中包含了两个张量（tensors）。这些张量表示链接的两个端点的偏移值，通过对链接数据的行和列减去节点偏移量来计算得到。
创建节点数量字典 node_count：

遍历了输入的节点计数信息（在 dl.nodes['count'] 中），并将每个节点类型的计数信息存储在 node_count 字典中。
构建异构图对象 G：

使用 dgl.heterograph 创建了异构图。这个函数以 edge_dict 为边的信息，node_count 为节点的数量信息创建了一个 DGL（Deep Graph Library） 异构图对象。
为了构建图对象，还需要提供 num_nodes_dict 参数来指定不同类型节点的数量。这里使用了 device 参数，可能用于将图对象分配到特定的计算设备上。
图节点和边的属性设置：

构建了 node_dict 和 edge_dict，这些字典包含了节点类型和边类型到数字 ID 的映射。
针对不同类型的边，通过设置每个边的 id 属性为相应类型的数字 ID，将边进行了标识。这种标识可能在图神经网络中的不同层或模型中有特定的作用。
返回创建的异构图对象 G：

最终，返回了构建好的异构图对象 G。
这段代码的主要目的是将输入的节点和边信息处理为一个 DGL 异构图对象，并为节点和边分配了特定的属性信息，以便于后续的图神经网络模型的构建和训练。'''
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def process_hgt(dl, g, args):
    edge_dict = {}

    for i, meta_path in dl.links['meta'].items():
        edge_dict[(str(meta_path[0]), str(meta_path[0]) + '_' + str(meta_path[1]), str(meta_path[1]))] = (torch.tensor(dl.links['data'][i].tocoo().row - dl.nodes['shift'][meta_path[0]]), torch.tensor(dl.links['data'][i].tocoo().col - dl.nodes['shift'][meta_path[1]]))

    node_count = {}
    for i, count in dl.nodes['count'].items():
        print(i, node_count)
        node_count[str(i)] = count

    G = dgl.heterograph(edge_dict, num_nodes_dict = node_count, device=device)
    """
    for ntype in G.ntypes:
        G.nodes[ntype].data['inp'] = dataset.nodes['attr'][ntype]
        # print(G.nodes['attr'][ntype].shape)
    """
    G.node_dict = {}
    G.edge_dict = {}
    for ntype in G.ntypes:
        G.node_dict[ntype] = len(G.node_dict)
    for etype in G.etypes:
        G.edge_dict[etype] = len(G.edge_dict)
        G.edges[etype].data['id'] = torch.ones(G.number_of_edges(etype), dtype=torch.long).to(device) * G.edge_dict[etype] 

    return G
'''
这段代码似乎是用于处理简单的 Heterogeneous Graph Neural Network（H-GNN）中的数据准备过程，主要涉及以下步骤：

process_edge2type 函数：

这个函数用于处理边到类型的映射关系，并返回一个字典 edge2type，表示每条边对应的类型。
根据不同的数据集（通过 args.dataset 参数），函数会根据链接信息构建边到类型的映射关系。
在构建过程中，会检查边的来源节点和目标节点对，根据数据中的链接信息来确定边的类型。
获取边特征 e_feat：

在函数主体中，通过调用 process_edge2type 函数获取边到类型的映射关系 edge2type。
遍历图 g 中的边（每对节点对），从 edge2type 字典中获取对应的边类型。
创建了一个列表 e_feat，其中每个元素表示对应边的类型。
最后将这些边类型转换为 torch.tensor 类型，并使用 to 方法将其移动到指定的 device（计算设备）上。
返回边特征 e_feat：

函数返回了表示边类型的张量 e_feat。
总体来说，这段代码主要用于从图的边信息中提取边的类型，将边类型作为图神经网络模型的输入特征。这种类型的特征可以帮助模型理解和处理异构图中不同类型的边，从而更好地进行后续的图神经网络训练或推理任务。'''
def process_simplehgnn(dl, g, args):
    def process_edge2type():
        edge2type = {}
        if args.dataset == 'IMDB':
            for k in dl.links['data']:
                for u, v in zip(*dl.links['data'][k].nonzero()):
                    edge2type[(u, v)] = k
            for i in range(dl.nodes['total']):
                edge2type[(i, i)] = len(dl.links['count'])
        else:
            for k in dl.links['data']:
                for u, v in zip(*dl.links['data'][k].nonzero()):
                    edge2type[(u, v)] = k 
            for i in range(dl.nodes['total']):
                if (i, i) not in edge2type:
                    edge2type[(i, i)] = len(dl.links['count'])
            for k in dl.links['data']:
                for u, v in zip(*dl.links['data'][k].nonzero()):
                    if (v, u) not in edge2type:
                        edge2type[(v, u)] = k + 1 + len(dl.links['count'])
        return edge2type
    
    e_feat = None
    edge2type = process_edge2type()
    e_feat = []
    for u, v in zip(*g.edges()):
        u = u.cpu().item()
        v = v.cpu().item()
        e_feat.append(edge2type[(u,v)])
    e_feat = torch.tensor(e_feat, dtype=torch.long).to(device)
    # logger.info(f"e_feat: {e_feat}")
    return e_feat

'''

这段代码定义了一个名为 process_magnn 的函数，它似乎是用于 MAGNN（Metapath Aggregated Graph Neural Network）中的数据处理。

函数内部执行以下操作：

获取参数 args 中的日志记录器 logger 和数据集名称 dataset_name。
定义了一个内部函数 get_adjlist_pkl，用于从输入的数据加载器 dl 中获取元路径对应的邻接表（Adjacency List）或者邻接列表的索引信息。
根据输入的元路径 meta，从数据加载器 dl 中获取元路径对应的稀疏矩阵信息。
根据稀疏矩阵的非零元素构建邻接表。
如果 return_dic 为 True，则返回邻接表列表 adjlist00 和一个包含元路径索引的字典 idx00；否则，将这些索引合并为一个数组并返回。
这段代码似乎是用于处理元路径信息，构建邻接表或索引，以便后续的模型训练或图数据处理。
总体来说，这个函数可能用于准备 MAGNN 模型所需的数据结构，包括元路径的邻接表或索引信息。'''
def process_magnn(args):
    logger = args.logger
    dataset_name = args.dataset
    '''
    
这段代码定义了一个名为 get_adjlist_pkl 的函数，它似乎用于处理图数据中的元路径（metapath）信息，主要完成以下几个任务：

获取元路径对应的邻接列表和索引：

函数接收了 dl（数据加载器）、meta（元路径）、type_id（节点类型ID）、return_dic（是否返回字典）、symmetric（是否对称）等参数。
使用 dl.get_meta_path(meta).tocoo() 获取元路径对应的稀疏矩阵。
根据稀疏矩阵的非零元素构建邻接列表。
对元路径进行逆序排列并转换为数组，创建元路径的索引信息。
如果 return_dic 为 True，则返回邻接列表 adjlist00 和元路径索引字典 idx00；否则将这些索引合并为一个数组并返回。
邻接列表构建：

首先，根据稀疏矩阵的非零元素构建了一个列表 adjlist00。
对于每个非零元素 (i, j, v)，将节点 i 与节点 j 的连接重复 v 次，并加入到 adjlist00 的对应位置。
最后，将列表中的连接关系转换为字符串格式。
元路径索引构建：

为了索引元路径信息，对元路径进行了逆序排列，并将其转换为一个数组。
为了构建索引字典 idx00，按元路径进行排序并以元组的形式存储。
返回结果：

函数返回邻接列表和元路径索引，根据 return_dic 参数的不同，返回类型可能是列表和字典的组合，或者仅仅是一个数组。
总体来说，这个函数似乎是用于构建图数据中元路径的邻接列表和索引，为后续的图神经网络模型的构建提供了对应的数据结构。'''
    
    def get_adjlist_pkl(dl, meta, type_id=0, return_dic=True, symmetric=False):
        meta010 = dl.get_meta_path(meta).tocoo()
        adjlist00 = [[] for _ in range(dl.nodes['count'][type_id])]
        for i, j, v in zip(meta010.row, meta010.col, meta010.data):
            adjlist00[i - dl.nodes['shift'][type_id]].extend([j - dl.nodes['shift'][type_id]] * int(v))
        adjlist00 = [' '.join(map(str, [i] + sorted(x))) for i, x in enumerate(adjlist00)]
        meta010 = dl.get_full_meta_path(meta, symmetric=symmetric)
        idx00 = {}
        for k in meta010:
            idx00[k] = np.array(sorted([tuple(reversed(i)) for i in meta010[k]]), dtype=np.int32).reshape([-1, len(meta)+1])
        if not return_dic:
            idx00 = np.concatenate(list(idx00.values()), axis=0)
        # logger.info(f"type_id: {type_id}, idx00: {idx00}")
        return adjlist00, idx00
    '''
    这段代码定义了一个名为 load_DBLP_data() 的函数，用于加载 DBLP 数据集，并对其进行预处理。

这个函数执行了以下操作：

使用 data_loader 从路径 'data/DBLP' 中加载数据集。
调用 get_adjlist_pkl 函数三次，分别处理三种不同的元路径，并获取它们对应的邻接列表和索引信息 (adjlist00, idx00, adjlist01, idx01, adjlist02, idx02)。
处理节点特征：
循环处理节点的特征信息，对于不存在特征的节点，使用单位矩阵表示。
将节点特征存储到 features_0, features_1, features_2, features_3 中。
处理邻接矩阵和节点类型：
将链接数据求和以获得邻接矩阵 adjM。
创建一个 type_mask 数组，用于标识节点类型。
处理标签：
对标签进行分割，并随机划分训练集和验证集。
使用 PAP（作者-论文-作者）关系定义论文之间的关系：
将邻接矩阵 adjM 转换为 PyTorch 张量，并进行一定的计算。
返回处理后的数据：
返回邻接列表、索引、节点特征、邻接矩阵、节点类型、标签以及训练/验证/测试集索引。
这个函数的主要目的似乎是为了加载和处理 DBLP 数据集，准备用于图神经网络模型的训练或其他图数据处理任务。

'''
    def load_DBLP_data():
        from utils.data_loader import data_loader
        dl = data_loader('data/DBLP')
        adjlist00, idx00 = get_adjlist_pkl(dl, [(0,1), (1,0)], symmetric=True)
        logger.info('meta path 1 done')
        adjlist01, idx01 = get_adjlist_pkl(dl, [(0,1), (1,2), (2,1), (1,0)], symmetric=True)
        logger.info('meta path 2 done')
        adjlist02, idx02 = get_adjlist_pkl(dl, [(0,1), (1,3), (3,1), (1,0)], symmetric=True)
        logger.info('meta path 3 done')
        
        # adjlist00, idx00 = get_adjlist_pkl(dl, [(0,1), (1,0)], return_dic=False, symmetric=True)
        # G00 = nx.readwrite.adjlist.parse_adjlist(adjlist00, create_using=nx.MultiDiGraph)
        # logger.info('meta path 1 done')
        # adjlist01, idx01 = get_adjlist_pkl(dl, [(0,1), (1,2), (2,1), (1,0)], return_dic=False, symmetric=True)
        # G01 = nx.readwrite.adjlist.parse_adjlist(adjlist01, create_using=nx.MultiDiGraph)
        # logger.info('meta path 2 done')
        # adjlist02, idx02 = get_adjlist_pkl(dl, [(0,1), (1,3), (3,1), (1,0)], return_dic=False, symmetric=True)
        # G02 = nx.readwrite.adjlist.parse_adjlist(adjlist02, create_using=nx.MultiDiGraph)
        # logger.info('meta path 3 done')
        
        features = []
        for i in range(4):
            th = dl.nodes['attr'][i]
            if th is None:
                features.append(np.eye(dl.nodes['count'][i], dtype=np.float32))
            else:
                if type(th) is np.ndarray:
                    features.append(th)
                else:
                    features.append(th.toarray())
        # 0 for authors, 1 for papers, 2 for terms, 3 for conferences
        features_0, features_1, features_2, features_3 = features

        adjM = sum(dl.links['data'].values())
        type_mask = np.zeros(dl.nodes['total'], dtype=np.int32)
        for i in range(4):
            type_mask[dl.nodes['shift'][i]: dl.nodes['shift'][i] + dl.nodes['count'][i]] = i
        labels = np.zeros((dl.nodes['count'][0], dl.labels_train['num_classes']), dtype=int)
        val_ratio = 0.2
        train_idx = np.nonzero(dl.labels_train['mask'])[0]
        np.random.shuffle(train_idx)
        split = int(train_idx.shape[0]*val_ratio)
        val_idx = train_idx[:split]
        train_idx = train_idx[split:]
        train_idx = np.sort(train_idx)
        val_idx = np.sort(val_idx)
        test_idx = np.nonzero(dl.labels_test['mask'])[0]
        labels[train_idx] = dl.labels_train['data'][train_idx]
        labels[val_idx] = dl.labels_train['data'][val_idx]
        labels = labels.argmax(axis=1)
        train_val_test_idx = {}
        train_val_test_idx['train_idx'] = train_idx
        train_val_test_idx['val_idx'] = val_idx
        train_val_test_idx['test_idx'] = test_idx
        
        # Using PAP to define relations between papers.
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        adjM = adjM.toarray()
        adjM = torch.FloatTensor(adjM).to(device)
        a_mask = np.where(type_mask == 0)[0]
        p_mask = np.where(type_mask == 1)[0]
        a_mask = torch.LongTensor(a_mask).to(device)
        p_mask = torch.LongTensor(p_mask).to(device)
        adjM[p_mask, :][:, p_mask] = torch.mm(adjM[p_mask, :][:, a_mask], adjM[a_mask, :][:, p_mask])
        adjM = adjM.data.cpu().numpy()
        torch.cuda.empty_cache()

        return [adjlist00, adjlist01, adjlist02], \
            [idx00, idx01, idx02], \
            [features_0, features_1, features_2, features_3],\
            adjM, \
            type_mask,\
            labels,\
            train_val_test_idx,\
            dl

        # return [[G00, G01, G02]], \
        #     [[idx00, idx01, idx02]], \
        #     [features_0, features_1, features_2, features_3],\
        #     adjM, \
        #     type_mask,\
        #     labels,\
        #     train_val_test_idx,\
        #     dl
    '''
    这个 load_ACM_data() 函数主要用于加载 ACM 数据集并进行处理。让我们来解释一下这个函数的主要步骤：

从指定路径 'data/ACM' 中使用 data_loader 加载数据集。
通过调用 dl.get_sub_graph([0, 1, 2]) 获取子图数据。
对每种类型的节点进行处理：
如果节点没有链接到其他节点，则将其连接到自身。
调用 get_adjlist_pkl 函数六次，分别处理六种不同的元路径，并获取它们对应的邻接列表和索引信息 (adjlist00, idx00, adjlist01, idx01, adjlist02, idx02, adjlist03, idx03, adjlist04, idx04, adjlist05, idx05)。
处理节点特征：
将节点特征存储到 features 中。
处理邻接矩阵和节点类型：
将链接数据求和以获得邻接矩阵 adjM。
创建一个 type_mask 数组，用于标识节点类型。
处理标签：
对标签进行分割，并随机划分训练集和验证集。
返回处理后的数据：
返回邻接列表、索引、节点特征、邻接矩阵、节点类型、标签以及训练/验证/测试集索引。
这个函数的目的是为了加载和处理 ACM 数据集，准备用于图神经网络模型的训练或其他图数据处理任务。'''
    def load_ACM_data():
        from utils.data_loader import data_loader
        dl = data_loader('data/ACM')
        dl.get_sub_graph([0, 1, 2])
        #dl.links['data'][0] += sp.eye(dl.nodes['total'])
        for i in range(dl.nodes['count'][0]):
            if dl.links['data'][0][i].sum() == 0:
                dl.links['data'][0][i,i] = 1
            if dl.links['data'][1][i].sum() == 0:
                dl.links['data'][1][i,i] = 1
        adjlist00, idx00 = get_adjlist_pkl(dl, [(0,1), (1,0)], symmetric=True)
        logger.info('meta path 1 done')
        adjlist01, idx01 = get_adjlist_pkl(dl, [(0,2), (2,0)], symmetric=True)
        logger.info('meta path 2 done')
        adjlist02, idx02 = get_adjlist_pkl(dl, [0, (0,1), (1,0)])
        logger.info('meta path 3 done')
        adjlist03, idx03 = get_adjlist_pkl(dl, [0, (0,2), (2,0)])
        logger.info('meta path 4 done')
        adjlist04, idx04 = get_adjlist_pkl(dl, [1, (0,1), (1,0)])
        logger.info('meta path 5 done')
        adjlist05, idx05 = get_adjlist_pkl(dl, [1, (0,2), (2,0)])
        logger.info('meta path 6 done')
        # adjlist00, idx00 = get_adjlist_pkl(dl, [(0,1), (1,0)], return_dic=False, symmetric=True)
        # G00 = nx.readwrite.adjlist.parse_adjlist(adjlist00, create_using=nx.MultiDiGraph)
        # logger.info('meta path 1 done')
        # adjlist01, idx01 = get_adjlist_pkl(dl, [(0,2), (2,0)], return_dic=False, symmetric=True)
        # G01= nx.readwrite.adjlist.parse_adjlist(adjlist01, create_using=nx.MultiDiGraph)
        # logger.info('meta path 2 done')
        # adjlist02, idx02 = get_adjlist_pkl(dl, [0, (0,1), (1,0)], return_dic=False)
        # G02 = nx.readwrite.adjlist.parse_adjlist(adjlist02, create_using=nx.MultiDiGraph)
        # logger.info('meta path 3 done')
        # adjlist03, idx03 = get_adjlist_pkl(dl, [0, (0,2), (2,0)], return_dic=False)
        # G03 = nx.readwrite.adjlist.parse_adjlist(adjlist03, create_using=nx.MultiDiGraph)
        # logger.info('meta path 4 done')
        # adjlist04, idx04 = get_adjlist_pkl(dl, [1, (0,1), (1,0)], return_dic=False)
        # G04 = nx.readwrite.adjlist.parse_adjlist(adjlist04, create_using=nx.MultiDiGraph)
        # logger.info('meta path 5 done')
        # adjlist05, idx05 = get_adjlist_pkl(dl, [1, (0,2), (2,0)], return_dic=False)
        # G05 = nx.readwrite.adjlist.parse_adjlist(adjlist05, create_using=nx.MultiDiGraph)
        # logger.info('meta path 6 done')
        features = []
        types = len(dl.nodes['count'])
        for i in range(types):
            th = dl.nodes['attr'][i]
            if th is None:
                features.append(np.eye(dl.nodes['count'][i], dtype=np.float32))
            else:
                if type(th) is np.ndarray:
                    features.append(th)
                else:
                    features.append(th.toarray())
        #features_0, features_1, features_2, features_3 = features

        adjM = sum(dl.links['data'].values())
        adjM = adjM.toarray()

        type_mask = np.zeros(dl.nodes['total'], dtype=np.int32)
        for i in range(types):
            type_mask[dl.nodes['shift'][i]:dl.nodes['shift'][i]+dl.nodes['count'][i]] = i
        labels = np.zeros((dl.nodes['count'][0], dl.labels_train['num_classes']), dtype=int)
        val_ratio = 0.2
        train_idx = np.nonzero(dl.labels_train['mask'])[0]
        np.random.shuffle(train_idx)
        split = int(train_idx.shape[0]*val_ratio)
        val_idx = train_idx[:split]
        train_idx = train_idx[split:]
        train_idx = np.sort(train_idx)
        val_idx = np.sort(val_idx)
        test_idx = np.nonzero(dl.labels_test['mask'])[0]
        labels[train_idx] = dl.labels_train['data'][train_idx]
        labels[val_idx] = dl.labels_train['data'][val_idx]
        labels = labels.argmax(axis=1)
        train_val_test_idx = {}
        train_val_test_idx['train_idx'] = train_idx
        train_val_test_idx['val_idx'] = val_idx
        train_val_test_idx['test_idx'] = test_idx

        return [adjlist00, adjlist01, adjlist02, adjlist03, adjlist04, adjlist05], \
                [idx00, idx01, idx02, idx03, idx04, idx05], \
                features, \
                adjM, \
                type_mask,\
                labels,\
                train_val_test_idx,\
                dl

        # return [[G00, G01, G02, G03, G04, G05]], \
        #         [[idx00, idx01, idx02, idx03, idx04, idx05]], \
        #         features, \
        #         adjM, \
        #         type_mask,\
        #         labels,\
        #         train_val_test_idx,\
        #         dl
    '''
    
load_IMDB_data() 函数主要用于处理 IMDB 数据集。以下是函数的主要步骤：

使用 data_loader 从 'data/IMDB' 路径加载 IMDB 数据集。
使用 get_adjlist_pkl 函数六次，分别处理六种不同的元路径，并获取它们对应的邻接列表和索引信息 (adjlist00, idx00, adjlist01, idx01, adjlist10, idx10, adjlist11, idx11, adjlist20, idx20, adjlist21, idx21)。
处理节点特征：
将节点特征存储到 features 中。
处理邻接矩阵和节点类型：
将链接数据求和以获得邻接矩阵 adjM。
创建一个 type_mask 数组，用于标识节点类型。
处理标签：
对标签进行分割，并随机划分训练集和验证集。
返回处理后的数据：
返回邻接图列表、索引列表、节点特征、邻接矩阵、节点类型、标签以及训练/验证/测试集索引。
此函数的目的是为了加载和处理 IMDB 数据集，准备用于图神经网络模型的训练或其他图数据处理任务。'''
    def load_IMDB_data():
        from utils.data_loader import data_loader
        dl = data_loader('data/IMDB')
        adjlist00, idx00 = get_adjlist_pkl(dl, [(0,1), (1,0)], 0, False, True)
        G00 = nx.readwrite.adjlist.parse_adjlist(adjlist00, create_using=nx.MultiDiGraph)
        logger.info('meta path 1 done')
        adjlist01, idx01 = get_adjlist_pkl(dl, [(0,2), (2,0)], 0, False, True)
        G01 = nx.readwrite.adjlist.parse_adjlist(adjlist01, create_using=nx.MultiDiGraph)
        logger.info('meta path 2 done')
        adjlist10, idx10 = get_adjlist_pkl(dl, [(1,0), (0,1)], 1, False, True)
        G10 = nx.readwrite.adjlist.parse_adjlist(adjlist10, create_using=nx.MultiDiGraph)
        logger.info('meta path 3 done')
        adjlist11, idx11 = get_adjlist_pkl(dl, [(1,0), (0,2), (2,0), (0, 1)], 1, False, True)
        G11 = nx.readwrite.adjlist.parse_adjlist(adjlist11, create_using=nx.MultiDiGraph)
        logger.info('meta path 4 done')
        adjlist20, idx20 = get_adjlist_pkl(dl, [(2,0), (0,2)], 2, False, True)
        G20 = nx.readwrite.adjlist.parse_adjlist(adjlist20, create_using=nx.MultiDiGraph)
        logger.info('meta path 5 done')
        adjlist21, idx21 = get_adjlist_pkl(dl, [(2,0), (0,1), (1,0), (0,2)], 2, False, True)
        G21 = nx.readwrite.adjlist.parse_adjlist(adjlist21, create_using=nx.MultiDiGraph)
        logger.info('meta path 6 done')
        features = []
        types = len(dl.nodes['count'])
        for i in range(types):
            th = dl.nodes['attr'][i]
            if th is None:
                features.append(np.eye(dl.nodes['count'][i], dtype=np.float32))
            else:
                if type(th) is np.ndarray:
                    features.append(th)
                else:
                    features.append(th.toarray())
        #features_0, features_1, features_2, features_3 = features

        adjM = sum(dl.links['data'].values())
        type_mask = np.zeros(dl.nodes['total'], dtype=np.int32)
        for i in range(types):
            type_mask[dl.nodes['shift'][i]:dl.nodes['shift'][i]+dl.nodes['count'][i]] = i
        labels = np.zeros((dl.nodes['count'][0], dl.labels_train['num_classes']), dtype=int)
        val_ratio = 0.2
        train_idx = np.nonzero(dl.labels_train['mask'])[0]
        np.random.shuffle(train_idx)
        split = int(train_idx.shape[0]*val_ratio)
        val_idx = train_idx[:split]
        train_idx = train_idx[split:]
        train_idx = np.sort(train_idx)
        val_idx = np.sort(val_idx)
        test_idx = np.nonzero(dl.labels_test['mask'])[0]
        labels[train_idx] = dl.labels_train['data'][train_idx]
        labels[val_idx] = dl.labels_train['data'][val_idx]
        #labels = labels.argmax(axis=1)
        train_val_test_idx = {}
        train_val_test_idx['train_idx'] = train_idx
        train_val_test_idx['val_idx'] = val_idx
        train_val_test_idx['test_idx'] = test_idx

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        adjM = adjM.toarray()
        adjM = torch.FloatTensor(adjM).to(device)
        m_mask = np.where(type_mask == 0)[0]
        d_mask = np.where(type_mask == 1)[0]
        a_mask = np.where(type_mask == 2)[0]
        a_mask = torch.LongTensor(a_mask).to(device)
        m_mask = torch.LongTensor(m_mask).to(device)
        d_mask = torch.LongTensor(d_mask).to(device)

        adjM[m_mask, :][:, m_mask] = torch.mm(adjM[m_mask, :][:, a_mask], adjM[a_mask, :][:, m_mask])
        adjM[m_mask, :][:, m_mask] = adjM[m_mask, :][:, m_mask] + torch.mm(adjM[m_mask, :][:, d_mask], adjM[d_mask, :][:, m_mask])
        adjM = adjM.data.cpu().numpy()
        torch.cuda.empty_cache()

        return [[G00, G01], [G10, G11], [G20, G21]], \
                [[idx00, idx01], [idx10, idx11], [idx20, idx21]], \
                features, \
                adjM, \
                type_mask,\
                labels,\
                train_val_test_idx,\
                dl
    '''
    这段代码是根据不同的数据集名字加载不同的数据。根据 dataset_name 的不同取值，会调用不同的数据加载函数来处理数据集。在每个 if 语句中，会调用相应的数据加载函数，并返回处理后的数据。

如果 dataset_name 是 'DBLP'，则会调用 load_DBLP_data() 函数来处理 DBLP 数据集，并返回得到的邻接列表和边元路径索引列表。
如果 dataset_name 是 'ACM'，则会调用 load_ACM_data() 函数来处理 ACM 数据集，并返回得到的邻接列表和边元路径索引列表。
如果 dataset_name 是 'IMDB'，则会调用 load_IMDB_data() 函数来处理 IMDB 数据集，并返回得到的邻接图列表、边元路径索引列表。
在处理 IMDB 数据集时，还会将索引列表转换为 PyTorch 张量，并将其移动到适当的设备上。

这段代码的目的是根据不同的数据集名称来加载不同的数据，并将处理后的数据返回给调用者。'''
    if dataset_name == 'DBLP':
        adjlists, edge_metapath_indices_list, features_list, adjM, type_mask, labels, train_val_test_idx, dl = load_DBLP_data()
        return (adjlists, edge_metapath_indices_list)
    elif dataset_name == 'ACM':
        adjlists, edge_metapath_indices_list, features_list, adjM, type_mask, labels, train_val_test_idx, dl = load_ACM_data()
        return (adjlists, edge_metapath_indices_list)
    elif dataset_name == 'IMDB':
        nx_G_lists, edge_metapath_indices_lists, features_list, adjM, type_mask, labels, train_val_test_idx, dl = load_IMDB_data()
        edge_metapath_indices_lists = [[torch.LongTensor(indices).to(device) for indices in indices_list] for indices_list in
                                    edge_metapath_indices_lists]

        '''
        这段代码看起来是将 NetworkX 图对象列表转换为 DGL 图对象列表，并将其移动到特定设备上。具体步骤如下：

创建一个空列表 g_lists，用于存储转换后的 DGL 图对象列表。
对于给定的 nx_G_lists（这可能是 NetworkX 图对象的列表列表），遍历其中的每个列表 nx_G_list。
对于每个 nx_G（NetworkX 图对象），创建一个新的 DGL 图对象 g，设置为允许多重边（multigraph=True）。
向 g 中添加节点，节点数量等于 nx_G 的节点数量。
向 g 中添加边，使用 nx_G.edges() 获取边的列表，并对边进行排序和转换为整数后添加到 DGL 图中。
将转换后的 DGL 图对象添加到 g_lists 列表中，并将这些图对象移动到指定的设备上（使用 .to(device)）。
最后返回包含 DGL 图对象列表和边元路径索引列表的元组 (g_lists, edge_metapath_indices_lists)。
这段代码的目的是将 NetworkX 图对象列表转换为 DGL 图对象列表，并将它们移到指定的设备上。'''
        g_lists = []
        for nx_G_list in nx_G_lists:
            g_lists.append([])
            for nx_G in nx_G_list:
                g = dgl.DGLGraph(multigraph=True)
                g.add_nodes(nx_G.number_of_nodes())
                g.add_edges(*list(zip(*sorted(map(lambda tup: (int(tup[0]), int(tup[1])), nx_G.edges())))))
                g_lists[-1].append(g.to(device))
        return (g_lists, edge_metapath_indices_lists)
    
    # if dataset_name == 'DBLP':
    #     nx_G_lists, edge_metapath_indices_lists, features_list, adjM, type_mask, labels, train_val_test_idx, dl = load_DBLP_data()
    # elif dataset_name == 'ACM':
    #     nx_G_lists, edge_metapath_indices_lists, features_list, adjM, type_mask, labels, train_val_test_idx, dl = load_ACM_data()
    # elif dataset_name == 'IMDB':
    #     nx_G_lists, edge_metapath_indices_lists, features_list, adjM, type_mask, labels, train_val_test_idx, dl = load_IMDB_data()
    
            
    # edge_metapath_indices_lists = [[torch.LongTensor(indices).to(device) for indices in indices_list] for indices_list in
    #                             edge_metapath_indices_lists]
    # g_lists = []
    # for nx_G_list in nx_G_lists:
    #     g_lists.append([])
    #     for nx_G in nx_G_list:
    #         g = dgl.DGLGraph(multigraph=True)
    #         g.add_nodes(nx_G.number_of_nodes())
    #         g.add_edges(*list(zip(*sorted(map(lambda tup: (int(tup[0]), int(tup[1])), nx_G.edges())))))
    #         g_lists[-1].append(g.to(device))
    # return (g_lists, edge_metapath_indices_lists)
