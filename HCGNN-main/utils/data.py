import os
import networkx as nx
import numpy as np
import scipy
import pickle
import scipy.sparse as sp

def load_data(prefix='DBLP'):
    from .data_loader import data_loader
    dl = data_loader('data/' + prefix)#这里加载了数据加载的信息，节点集，边集，标签训练集、标签测试集
    # print(os.path.abspath('../../data/' + prefix))

    # dl = data_loader('data/'+prefix)
    features = []#这里便得到了所有的节点属性
    for i in range(len(dl.nodes['count'])):
        th = dl.nodes['attr'][i]#这里获取节点各类的属性（4057,334）、（14328,4231）、（7723,50）
        if th is None:
            features.append(sp.eye(dl.nodes['count'][i]))
        else:
            features.append(th)

    adjM = sum(dl.links['data'].values())
    type_mask = np.zeros(dl.nodes['total'], dtype=np.int32)
    
    node_type_num = len(dl.nodes['count'])#这一步获取了节点的不同类型总数，，然后在dl.nodes['count']记录了不同类型节点的各个数量
    
    for i in range(node_type_num):#for i in range（4）：由左闭右开原则，i的结果为0、1、2、3，这里是为了区分不同类型的节点
        type_mask[dl.nodes['shift'][i]: dl.nodes['shift'][i]+dl.nodes['count'][i]] = i
        
    labels = np.zeros((dl.nodes['count'][0], dl.labels_train['num_classes']), dtype=int)#这样的矩阵用于存储节点的标签信息
    val_ratio = 0.2#验证率设为0.2
    train_idx = np.nonzero(dl.labels_train['mask'])[0]
    np.random.shuffle(train_idx)
    split = int(train_idx.shape[0] * val_ratio)#1217*0.2=243
    val_idx = train_idx[:split]#这个操作通常用于将数据集划分成训练集和验证集。通过使用 train_idx 中找到的索引，从训练集中选择了一部分用作验证集。
    train_idx = train_idx[split:]
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)
    test_idx = np.nonzero(dl.labels_test['mask'])[0]
    labels[train_idx] = dl.labels_train['data'][train_idx]
    labels[val_idx] = dl.labels_train['data'][val_idx]
    if prefix != 'IMDB':
        labels = labels.argmax(axis=1)
    train_val_test_idx = {}
    train_val_test_idx['train_idx'] = train_idx
    train_val_test_idx['val_idx'] = val_idx
    train_val_test_idx['test_idx'] = test_idx
    # emb = np.load('D:/pycharm_item/AUTOAC/AutoAC-main/data/' + prefix + '/metapath2vec_emb.npy')
    # emb = np.load('/home/yyj/MDNN-AC/AutoAC-main/data/' + prefix + '/metapath2vec_emb.npy')
    return features,\
        adjM, \
        type_mask, \
        labels,\
        train_val_test_idx,\
        dl#,emb

'''
这段代码主要是一个用于加载异质图数据的函数 load_data。以下是这个函数执行的一些操作：
该函数假设数据文件位于名为 DBLP 的文件夹内（通过参数 prefix 指定，默认为 'DBLP'）。
使用 data_loader 从数据文件中加载数据。具体实现位于 'data_loader.py' 文件中。
循环遍历节点，并根据节点的属性（如果存在）创建特征矩阵列表 features。
合并连接边，生成邻接矩阵 adjM。
创建节点类型掩码 type_mask，用于标识每个节点所属的节点类型。
处理标签数据：
将训练标签的一部分作为验证集，剩余部分作为训练集。
对于不同的数据集（非 'IMDB' 数据集），对标签进行处理。
将数据整理为特征、邻接矩阵、节点类型掩码、标签、训练/验证/测试索引等数据结构，并返回给调用者。
此函数的作用是加载数据并对其进行预处理，以便后续用于图神经网络或其他机器学习任务中。
'''


# # 计算MRR
# def mean_reciprocal_rank(y_true, pred):
#     sorted_indices = np.argsort(pred)[::-1]  # 按预测概率降序排列
#     true_indices = np.where(y_true == 1)[0]  # 真实正例的索引
#     reciprocal_ranks = []
#     for idx in true_indices:
#         rank = np.where(sorted_indices == idx)[0]  # 找到正例的排名
#         reciprocal_rank = 1 / (rank + 1) if rank.size > 0 else 0  # 计算倒数排名
#         reciprocal_ranks.append(reciprocal_rank)
#     return np.mean(reciprocal_ranks)