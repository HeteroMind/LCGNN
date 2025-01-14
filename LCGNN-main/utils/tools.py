import os
import numpy as np
import scipy.sparse as sp
import torch
import numpy as np
import logging
import dgl
import sys

from ops.operations import *
from itertools import combinations
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

'''
这段代码定义了一个函数，用于获取日志记录器对象。它首先创建一个日志记录器对象，然后根据参数设置日志记录级别和输出格式。
如果指定了保存目录，则创建文件处理器和流处理器，并将它们添加到日志记录器中。
最后返回日志记录器对象。在最后一行代码中，调用 get_logger 函数并将返回的日志记录器对象赋值给变量 logger。'''
def get_logger(name=__file__, save_dir=None, level=logging.INFO):
    
    logger = logging.getLogger(name)

    if save_dir is None:
        return logger
    
    if getattr(logger, '_init_done__', None):
        logger.setLevel(level)
        return logger

    logger._init_done__ = True
    logger.propagate = False
    logger.setLevel(level)

    # logging.basicConfig(stream=sys.stdout, level=level, format=log_fmt, datefmt=date_fmt)
    
    formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
    # handler = logging.StreamHandler()
        
    shandler = logging.StreamHandler()
    shandler.setFormatter(formatter)
    shandler.setLevel(0)

    handler = logging.FileHandler(save_dir)
    handler.setFormatter(formatter)
    # handler.setLevel(0)

    del logger.handlers[:]
    logger.addHandler(handler)
    logger.addHandler(shandler)

    return logger

logger = get_logger()
'''
这段代码定义了一个函数，用于将稀疏矩阵转换为 PyTorch 的稀疏张量。它首先将稀疏矩阵转换为 COO 格式的稀疏矩阵，然后使用 PyTorch 的 torch.sparse.FloatTensor 函数创建稀疏张量，并指定索引、值和形状。最后返回创建的稀疏张量。'''
def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))
'''

这段代码定义了一个函数，用于将矩阵转换为 PyTorch 张量。如果输入是 NumPy 数组，则使用 torch.from_numpy 函数将其转换为张量并指定数据类型为浮点型；否则调用 sp_to_spt 函数进行转换。'''
def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)
'''

这段代码定义了一个函数，用于将 NumPy 数组转换为 PyTorch 张量。它首先遍历输入的特征列表，如果特征是 NumPy 数组，则调用 mat2tensor 函数将其转换为张量并移动到指定的设备上，然后将处理后的特征添加到输入列表中。接着根据数据集类型创建目标张量，并将其移动到指定的设备上。最后返回处理后的输入列表和目标张量。'''
def convert_np2torch(X, y, args, y_idx=None):
    input = []
    for features in X:
        if type(features) is np.ndarray:
            input.append(mat2tensor(features).to(device))
        else:
            input.append(features)
    # if type(X[0]) is np.ndarray:
    #     # logger.info(f"type(X) is np.ndarray")
    #     input = [mat2tensor(features).to(device) for features in X]
    # else:
    #     # logger.info(f"type(X) is not np.ndarray")
    #     input = X
    _y = y[y_idx] if y_idx is not None else y
    if args.dataset == 'IMDB':
        target = torch.FloatTensor(_y).to(device)    
    else:
        target = torch.LongTensor(_y).to(device)
    
    return input, target

# def get_logger(name=__file__, level=logging.INFO):
#     logger = logging.getLogger(name)

#     if getattr(logger, '_init_done__', None):
#         logger.setLevel(level)
#         return logger

#     logger._init_done__ = True
#     logger.propagate = False
#     logger.setLevel(level)

#     formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
#     handler = logging.StreamHandler()
#     handler.setFormatter(formatter)
#     handler.setLevel(0)

#     del logger.handlers[:]
#     logger.addHandler(handler)

#     return logger

# logger = get_logger()

primitives_str = '.'.join(PRIMITIVES)

# logger.info(primitives_str)
'''
这段代码定义了一个函数，用于将列表形式的稀疏矩阵转换为稀疏矩阵对象。它首先从列表中提取出数据、行索引和列索引，然后使用 scipy 库的 coo_matrix 函数创建稀疏矩阵，并指定形状为给定的矩阵形状。最后返回创建的稀疏矩阵对象。'''
def list_to_sp_mat(li, matrix_shape):
    n, m = matrix_shape
    data = [x[2] for x in li]
    i = [x[0] for x in li]
    j = [x[1] for x in li]
    return sp.coo_matrix((data, (i, j)), shape=(n, m))# .tocsr()

'''
这段代码定义了一个函数，用于将稀疏矩阵转换为 PyTorch 的稀疏张量。它首先将稀疏矩阵转换为 COO 格式的稀疏矩阵，然后使用 PyTorch 的 torch.sparse_coo_tensor 函数创建稀疏张量，并指定设备为给定的设备。最后返回创建的稀疏张量。'''
def to_torch_sp_mat(li, matrix_shape, device):
    n, m = matrix_shape
    sp_info = list_to_sp_mat(li, matrix_shape)
    
    values = sp_info.data
    # print(f"sp_info: {sp_info}")
    # print(f"values: {values}")

    indices = np.vstack((sp_info.row, sp_info.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = sp_info.shape

    # torch_sp_mat = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    torch_sp_mat = torch.sparse_coo_tensor(i, v, torch.Size(shape), device=device)

    return torch_sp_mat
'''
这段代码定义了一个函数，用于判断新的中心点是否与之前的中心点足够接近。它首先计算中心点之间的距离，然后判断是否有任何距离大于给定的阈值 eps，如果有则返回 False 和平均距离值，否则返回 True 和平均距离值。'''
def is_center_close(prev_centers, new_centers, eps):
    temp_gap = new_centers - prev_centers
    gap_value = (temp_gap ** 2).sum(axis=1)
    gap_value_avg = np.mean(gap_value)
    for x in gap_value:
        if x >= eps:
            return False, gap_value_avg
    return True, gap_value_avg
'''!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
这段代码定义了一个函数，用于根据给定的索引将节点特征进行替换。它首先判断是否存在初始节点特征，然后根据给定的索引和节点特征进行替换操作，并返回替换后的节点特征。'''
def scatter_embbeding(_node_embedding, h_attribute, node_embedding, idx_batch): 
    if _node_embedding is None:
        _node_embedding = h_attribute
    # logger.info(f"_node_embedding shape: {_node_embedding.shape}; node_embedding shape: {node_embedding.shape}")
    # logger.info(f"idx_batch: {idx_batch}")
    tmp_id = 0
    for row_idx in idx_batch:
        # logger.info(f"_node_embedding[row_idx]: {_node_embedding[row_idx]}")
        _node_embedding[row_idx] = node_embedding[tmp_id]
        tmp_id += 1
    return _node_embedding
'''
这段代码定义了一个函数，用于将节点特征按照给定的索引进行累加。它首先判断是否存在初始节点特征，然后根据给定的索引和节点特征进行累加操作，并返回累加后的节点特征。'''
def scatter_add(_node_embedding, h_attribute, node_embedding, idx_batch): 
    if _node_embedding is None:
        _node_embedding = h_attribute
    # logger.info(f"_node_embedding shape: {_node_embedding.shape}; node_embedding shape: {node_embedding.shape}")
    # logger.info(f"idx_batch: {idx_batch}")
    n = len(idx_batch)
    m = h_attribute.shape[1]
    index = torch.LongTensor(idx_batch).cuda()
    _node_embedding.index_fill(0, index, 0)
    
    index_exp = torch.unsqueeze(index, 1)
    index_exp = index_exp.expand(n, m)
    _node_embedding.scatter_add_(0, index_exp, node_embedding)

    return _node_embedding

'''
这段代码定义了一个用于生成索引的类。它包含了初始化函数用于设置参数，以及一些方法用于生成索引、计算迭代次数和重置迭代器。'''
class index_generator:
    def __init__(self, batch_size, num_data=None, indices=None, shuffle=True):
        if num_data is not None:
            self.num_data = num_data
            self.indices = np.arange(num_data)
        if indices is not None:
            self.num_data = len(indices)
            self.indices = np.copy(indices)
        self.batch_size = batch_size
        self.iter_counter = 0
        self.shuffle = shuffle
        if shuffle:
            np.random.shuffle(self.indices)

    def next(self):
        if self.num_iterations_left() <= 0:
            self.reset()
        self.iter_counter += 1
        return np.copy(self.indices[(self.iter_counter - 1) * self.batch_size:self.iter_counter * self.batch_size])

    def num_iterations(self):
        return int(np.ceil(self.num_data / self.batch_size))

    def num_iterations_left(self):
        return self.num_iterations() - self.iter_counter

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.iter_counter = 0
'''
这段代码定义了一个函数，用于解析邻接列表和边元路径索引。它会根据输入的邻接列表和索引，解析出边的信息、结果索引、节点数量和映射关系，并将这些信息返回。'''
def parse_adjlist(adjlist, edge_metapath_indices, samples=None):
    edges = []
    nodes = set()
    result_indices = []
    for row, indices in zip(adjlist, edge_metapath_indices):
        row_parsed = list(map(int, row.split(' ')))
        nodes.add(row_parsed[0])
        if len(row_parsed) > 1:
            # sampling neighbors
            if samples is None:
                neighbors = row_parsed[1:]
                result_indices.append(indices)
            else:
                # undersampling frequent neighbors
                unique, counts = np.unique(row_parsed[1:], return_counts=True)
                p = []
                for count in counts:
                    p += [(count ** (3 / 4)) / count] * count
                p = np.array(p)
                p = p / p.sum()
                samples = min(samples, len(row_parsed) - 1)
                sampled_idx = np.sort(np.random.choice(len(row_parsed) - 1, samples, replace=False, p=p))
                neighbors = [row_parsed[i + 1] for i in sampled_idx]
                result_indices.append(indices[sampled_idx])
        else:
            neighbors = []
            result_indices.append(indices)
        for dst in neighbors:
            nodes.add(dst)
            edges.append((row_parsed[0], dst))
    mapping = {map_from: map_to for map_to, map_from in enumerate(sorted(nodes))}
    edges = list(map(lambda tup: (mapping[tup[0]], mapping[tup[1]]), edges))
    result_indices = np.vstack(result_indices)
    return edges, result_indices, len(nodes), mapping

'''这段代码定义了一个函数，用于解析小批量数据。它接受一些邻接列表、边元路径索引列表、批次索引等参数，然后根据这些参数解析出图结构和相应的结果索引。最后返回解析后的图列表、结果索引列表和映射后的批次索引列表。'''
def parse_minibatch(adjlists, edge_metapath_indices_list, idx_batch, device, samples=None):
    g_list = []
    result_indices_list = []
    idx_batch_mapped_list = []
    for adjlist, indices in zip(adjlists, edge_metapath_indices_list):
        edges, result_indices, num_nodes, mapping = parse_adjlist(
            [adjlist[i] for i in idx_batch], [indices[i] for i in idx_batch], samples)

        g = dgl.DGLGraph(multigraph=True)
        g.add_nodes(num_nodes)
        if len(edges) > 0:
            sorted_index = sorted(range(len(edges)), key=lambda i : edges[i])
            g.add_edges(*list(zip(*[(edges[i][1], edges[i][0]) for i in sorted_index])))
            result_indices = torch.LongTensor(result_indices[sorted_index]).to(device)
        else:
            result_indices = torch.LongTensor(result_indices).to(device)
        #g.add_edges(*list(zip(*[(dst, src) for src, dst in sorted(edges)])))
        #result_indices = torch.LongTensor(result_indices).to(device)
        g_list.append(g.to(device))
        result_indices_list.append(result_indices)
        idx_batch_mapped_list.append(np.array([mapping[idx] for idx in idx_batch]))

    return g_list, result_indices_list, idx_batch_mapped_list

'''
这段代码定义了另一个用于提前停止训练的类，与之前的类功能相同，用于监视验证损失的变化，如果在一定耐心值内验证损失没有改善，则会提前停止训练。'''
class EarlyStopping_Retrain:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, logger, patience, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """

        self._logger = logger
        self.patience = patience
        self.verbose = verbose
        self.counter = 0

        self.best_score_val = None

        self.early_stop = False
        self.val_loss_min = np.Inf
        self.train_loss_min = np.Inf
        self.delta = delta

    def __call__(self, train_loss, val_loss):

        score_val = -val_loss

        if self.best_score_val is None:
            self.best_score_val = score_val
        elif score_val < self.best_score_val - self.delta:
            self.counter += 1
            self._logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self._logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            if val_loss < self.val_loss_min:
                self.val_loss_min = val_loss
            self.best_score_val = score_val
            self.counter = 0

# class EarlyStopping_Retrain:
#     """Early stops the training if validation loss doesn't improve after a given patience."""
#     def __init__(self, args, patience, verbose=False, delta=0, save_path='checkpoint.pt'):
#         """
#         Args:
#             patience (int): How long to wait after last time validation loss improved.
#                             Default: 7
#             verbose (bool): If True, prints a message for each validation loss improvement.
#                             Default: False
#             delta (float): Minimum change in the monitored quantity to qualify as an improvement.
#                             Default: 0
#         """
#         self.args =  args
#         self.patience = patience
#         self.verbose = verbose
#         self.counter = 0

#         self.best_score_train = None
#         self.best_score_val = None

#         self.early_stop = False
#         self.val_loss_min = np.Inf
#         self.train_loss_min = np.Inf
#         self.delta = delta
#         self.save_path = save_path
        
#     def __call__(self, train_loss, val_loss, model):

#         score_train = -train_loss
#         score_val = -val_loss

#         if self.best_score_train is None:
#             self.best_score_train = score_train
#             self.best_score_val = score_val

#             self.save_checkpoint(train_loss, val_loss, model)
        
#         elif score_train < self.best_score_train - self.delta and score_val < self.best_score_val:
#         # elif score_val < self.best_score_val - self.delta:
#             self.counter += 1
#             logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             if score_train >= self.best_score_train:
#                 self.best_score_train = score_train
#             if score_val >= self.best_score_val:
#                 self.best_score_val = score_val
#             self.save_checkpoint(train_loss, val_loss, model)
#             self.counter = 0

#     def save_checkpoint(self, train_loss, val_loss, model):
#         """Saves model when validation loss decrease."""
#         if self.verbose:
#             if train_loss < self.train_loss_min and val_loss < self.val_loss_min:
#                 logger.info(f'Training loss decreased ({self.train_loss_min:.6f} --> {train_loss:.6f}). Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
#             elif train_loss < self.train_loss_min:
#                 logger.info(f'Training loss decreased ({self.train_loss_min:.6f} --> {train_loss:.6f}).  Saving model ...')
#             else:
#                 logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

#         torch.save(model.state_dict(), self.save_path)

#         # self.val_loss_min = val_loss
#         if train_loss < self.train_loss_min: self.train_loss_min = train_loss
#         if val_loss < self.val_loss_min: self.val_loss_min = val_loss
'''
这段代码定义了一个用于提前停止训练的类。它会监视验证损失的变化，如果在一定耐心值内验证损失没有改善，则会提前停止训练。'''
class EarlyStopping_Search:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, logger, patience, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        
        self._logger = logger
        self.patience = patience
        self.verbose = verbose
        self.counter = 0

        self.best_score_val = None

        self.early_stop = False
        self.val_loss_min = np.Inf
        self.train_loss_min = np.Inf
        self.delta = delta

    def __call__(self, train_loss, val_loss):

        score_val = -val_loss

        if self.best_score_val is None:
            self.best_score_val = score_val
        elif score_val < self.best_score_val - self.delta:
            self.counter += 1
            self._logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self._logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            if val_loss < self.val_loss_min:
                self.val_loss_min = val_loss
            self.best_score_val = score_val
            self.counter = 0



# class EarlyStopping_Search:
#     """Early stops the training if validation loss doesn't improve after a given patience."""
#     def __init__(self, args, patience, verbose=False, delta=0):
#         """
#         Args:
#             patience (int): How long to wait after last time validation loss improved.
#                             Default: 7
#             verbose (bool): If True, prints a message for each validation loss improvement.
#                             Default: False
#             delta (float): Minimum change in the monitored quantity to qualify as an improvement.
#                             Default: 0
#         """
#         self.args =  args
#         self.patience = patience
#         self.verbose = verbose
#         self.counter = 0

#         self.best_score_train = None
#         self.best_score_val = None

#         self.early_stop = False
#         self.val_loss_min = np.Inf
#         self.train_loss_min = np.Inf
#         self.delta = delta

#         # create dir
#         opt = 'SGD' if args.useSGD else 'Adam'
#         is_rolled = 'unrolled' if args.unrolled else 'one-prox'
#         is_use_type_linear = 'typeLinear' if args.useTypeLinear else 'noTypeLinear'
#         dir_name = args.dataset + '_' + 'C' + str(args.cluster_num) + '_' + args.gnn_model + \
#                     '_' + is_use_type_linear + \
#                     '_' + opt + '_' + is_rolled + '_epoch' + str(args.epoch) + \
#                     '_' + primitives_str + '_' + args.time_line
#         self.dir_name = dir_name
#         logger.info(f"save_dir_name: {dir_name}")
#         self.base_dir = os.path.join('disrete_arch_info', dir_name)

#         if not os.path.exists(self.base_dir):
#             os.makedirs(self.base_dir)

#     def save_name(self):
#         return self.dir_name

#     def __call__(self, train_loss, val_loss, node_assign, alpha_params):

#         score_train = -train_loss
#         score_val = -val_loss

#         if self.best_score_train is None:
#             self.best_score_train = score_train
#             self.best_score_val = score_val

#             self.save_checkpoint(train_loss, val_loss, node_assign, alpha_params)
        
#         elif score_train < self.best_score_train - self.delta and score_val < self.best_score_val:
#         # elif score_val < self.best_score_val - self.delta:
#             self.counter += 1
#             logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             if score_train >= self.best_score_train:
#                 self.best_score_train = score_train
#             if score_val >= self.best_score_val:
#                 self.best_score_val = score_val
#             self.save_checkpoint(train_loss, val_loss, node_assign, alpha_params)
#             self.counter = 0

#     def save_checkpoint(self, train_loss, val_loss, node_assign, alpha_params):
#         """Saves model when validation loss decrease."""
#         if self.verbose:
#             if train_loss < self.train_loss_min and val_loss < self.val_loss_min:
#                 logger.info(f'Training loss decreased ({self.train_loss_min:.6f} --> {train_loss:.6f}). Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
#             elif train_loss < self.train_loss_min:
#                 logger.info(f'Training loss decreased ({self.train_loss_min:.6f} --> {train_loss:.6f}).  Saving model ...')
#             else:
#                 logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

#         # if self.verbose:
#         #     logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

#         # print(os.path.abspath(self.save_path))
#         if node_assign is not None:
#             node_assign = np.array(node_assign)
#             node_file_name = os.path.join(self.base_dir, 'node_assign.npy')
#             np.save(node_file_name, node_assign)

#         alpha_file_name = os.path.join(self.base_dir, 'alpha_params.npy')
#         np.save(alpha_file_name, alpha_params)
        
#         # self.val_loss_min = val_loss
#         if train_loss < self.train_loss_min: self.train_loss_min = train_loss
#         if val_loss < self.val_loss_min: self.val_loss_min = val_loss
'''
这段代码是一个包含了数据转换、数据预处理和训练过程中的辅助函数和工具类。它实现了一些用于处理图数据、早期停止训练的机制和日志记录的函数。

其中的一些函数和类的作用包括：

get_logger: 用于创建一个日志记录器（logger）。
sp_to_spt: 将稀疏矩阵转换为 PyTorch 的稀疏张量。
mat2tensor: 将矩阵转换为 PyTorch 张量。
convert_np2torch: 将 Numpy 数组转换为 PyTorch 张量。
list_to_sp_mat: 将列表转换为稀疏矩阵。
to_torch_sp_mat: 将稀疏矩阵转换为 PyTorch 的稀疏张量，并在指定的设备上存储。
is_center_close: 检查两组中心点之间的距离是否小于给定的阈值。
scatter_embedding 和 scatter_add: 用于在节点嵌入中更新节点信息的函数。
index_generator: 生成用于迭代数据集的批次索引。
parse_adjlist 和 parse_minibatch: 用于解析邻接列表和生成 minibatch 数据的函数。
EarlyStopping_Retrain 和 EarlyStopping_Search: 用于在训练过程中实现早期停止的类，具体包括根据验证集损失或其他条件判断是否提前终止训练。
这些函数和类在处理图数据、构建模型、训练过程中都发挥了重要的作用，以及在训练过程中进行条件判断和终止等方面提供了一些机制。'''
#自己加的补全损失  以及  视角损失

import torch.nn.functional as F

# def completion_loss(guidance_loss_features):
def completion_loss(original_features_tensor, reconstructed_features_tensor,args):
    if args.dataset == 'IMDB':
          completion_loss = 1 - F.cosine_similarity(original_features_tensor.to(device), reconstructed_features_tensor.to(device), dim=1).sum() / \
             original_features_tensor.shape[0]
    else:
          completion_loss = F.pairwise_distance(original_features_tensor.to(device), reconstructed_features_tensor.to(device), 2).mean()

    # completion_loss = F.mse_loss(original_features_tensor.to(device), reconstructed_features_tensor.to(device))

    #completion_loss = mse_loss / original_features_tensor.shape[0]#所以这句可以取消了
    # # Initialize total loss as a tensor
    # total_loss = torch.tensor(0.0, device=device)
    # # Calculate the mean squared error for each node
    # for _, features_tuple in guidance_loss_features.items():
    #     #original_features, reconstructed_features = features_tuple
    #     #zengjiade
    #     # original_features_norm = F.normalize(original_features.to(device), p=2, dim=1)
    #     # reconstructed_features_norm = F.normalize(reconstructed_features.to(device), p=2, dim=1)
    #     #original_features_norm = F.normalize(original_features.to(device), p=2, dim=0)
    #     #reconstructed_features_norm = F.normalize(reconstructed_features.to(device), p=2, dim=0)
    #
    #     #mse_loss = F.mse_loss(original_features_norm.to(device), reconstructed_features_norm.to(device))
    #     mse_loss = F.mse_loss(features_tuple[0].to(device), features_tuple[1].to(device))
    #     total_loss += mse_loss.item()
    # # Calculate the completion loss using the formula
    # completion_loss = total_loss / len(guidance_loss_features)
    return completion_loss

# def consistency_loss(output):
#     # 计算余弦相似度矩阵
#     cos_similarity_matrices = []
#     for features in output:
#         # 计算标准化后的特征
#         normalized_features = F.normalize(features, p=2, dim=-1)
#         # 计算余弦相似度矩阵
#         cos_similarity_matrix = torch.matmul(normalized_features.T, normalized_features)
#         cos_similarity_matrices.append(cos_similarity_matrix)
#
#     # 计算一致性约束损失
#     consistency_loss = 0
#     num_views = len(cos_similarity_matrices)
#     for i in range(num_views):
#         for j in range(i+1, num_views):
#             # 计算差值矩阵的范数的平方并累加到损失中
#             consistency_loss += torch.norm(cos_similarity_matrices[i] - cos_similarity_matrices[j])**2
#
#     return consistency_loss
'''第一个能用的一致性损失'''
def calculate_cos_similarity_matrix(features):

    # 计算标准化后的特征
    normalized_features = torch.nn.functional.normalize(features, p=2, dim=-1).float().to(device)
    # 计算余弦相似度矩阵
    cos_similarity_matrix = torch.matmul(normalized_features.unsqueeze(1), normalized_features.unsqueeze(0)).to(device)
    return cos_similarity_matrix

def consistency_loss(output):
    # 计算各节点的余弦相似度矩阵
    cos_similarity_matrices = {}
    for node_id, features_list in output.items():#这是由于output中的features_list的数量不一致，所以就没用整个矩阵的形式去求
        cos_similarity_matrices[node_id] = []
        for features in features_list:
            cos_similarity_matrices[node_id].append(calculate_cos_similarity_matrix(features))

    # 计算一致性约束损失
    consistency_loss = torch.tensor(0.0, device=device)
    #num_pairs = 0#为了求平均
    for _, matrices_list in cos_similarity_matrices.items():
        num_views = len(matrices_list)
        for i in range(num_views):
            for j in range(i+1, num_views):
                # 计算差值矩阵的范数的平方并累加到损失中
                diff_matrix =matrices_list[i]- matrices_list[j]
                consistency_loss += (torch.norm(diff_matrix)**2).item()
                #num_pairs += 1#为了求平均
    # 求平均
    # if num_pairs > 0:
    #     consistency_loss /= num_pairs
    consistency_loss /= len(output)
    #直接给出
    #num_pairs = math.comb(num_views, 2)
    return consistency_loss
#原本能用的
def is_save(bst_val_loss, train_loss_classification, val_loss):
    if val_loss < bst_val_loss[0]:
        bst_val_loss[0] = val_loss
        return True
    return False

def is_save_1(recent_val_losses, val_loss):
    # 如果队列中元素已经满了（即已经收集了7个val_loss）
    if len(recent_val_losses) == 7:
        # 计算前七个val_loss的平均值
        average_val_loss = sum(recent_val_losses) / len(recent_val_losses)

        # 检查当前的val_loss（第8个及之后的）是否比之前七个val_loss的平均值要小
        if val_loss < average_val_loss:
            return True
    return False
def save_dir_name(args):

    dir_name = args.gnn_model + \
               '_' + 'lr' + str(args.lr) + \
               '_' + 'wd' + str(args.weight_decay)
    #'_' + primitives_str + '_'\
    dir_name = dir_name
    args.logger.info(f"save_dir_name: {dir_name}")

    return dir_name

def save_search_info(hgnn_model, args):
    # save_path_name = os.path.join('disrete_arch_info', save_dir_name(args))
    # if not os.path.exists(save_path_name):
    #     os.makedirs(save_path_name)
    #
    # save_path_name = save_path_name + '.npy'
    # #save_info = hgnn_model.state_dict()
    # save_info = get_checkpoint_info(hgnn_model,args)
    # np.save(save_path_name, save_info)
    #torch.save(hgnn_model.state_dict(), 'D:\\pycharm_item\\AUTOAC\\AutoAC-main\\checkpoint\\save\\net_params_{}.pt'.format(args.time_line))
    #torch.save(hgnn_model.state_dict(),'/home/yangyuanjun/MDNN-AC/AutoAC-main/checkpoint/save/net_params_{}.pt'.format(args.time_line))
    torch.save(hgnn_model.state_dict(),'/home/yyj/MDNN-AC/AutoAC-main/checkpoint/save/net_params_{}.pt'.format(args.time_line))

'''老师那篇论文迁移过来的损失'''

def loss_each_view(view_features):
    loss_com = 0
    com = combinations(view_features, 2)
    for x1, x2 in com:
        loss_com += common_loss(x1, x2)

    # # add L2 regularization
    # reg_loss = sum(torch.norm(x,p=2) for x in view_features)

    return loss_com #+0.01 * reg_loss

def common_loss(emb1, emb2):
    emb1 = emb1 - torch.mean(emb1, dim=0, keepdim=True)
    emb2 = emb2 - torch.mean(emb2, dim=0, keepdim=True)
    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)

    cov1 = torch.matmul(emb1, emb1.t())
    cov2 = torch.matmul(emb2, emb2.t())
    loss_com = torch.nn.MSELoss()
    com_loss = loss_com(cov1, cov2)

    # cos_sim = torch.matmul(emb1,emb2.t())
    # loss_com = torch.nn.MSELoss()
    # com_loss = loss_com(cos_sim,torch.eye(emb1.shape[0]).to(emb1.device))
    return com_loss

def Diversity_loss(view_features):
    loss_div = 0
    combin = combinations(view_features, 2)
    for x1, x2 in combin:
        loss_div += d_loss(x1, x2)

    return loss_div

def d_loss(emb1, emb2):
    similar = torch.bmm(emb1.unsqueeze(1),
                        emb2.unsqueeze(2))

    similar = torch.reshape(similar, shape=[emb1.shape[0]])
    norm_matrix_img = torch.norm(emb1, p=2, dim=1)
    norm_matrix_text = torch.norm(emb2, p=2, dim=1)
    div = torch.mean(similar / (norm_matrix_img * norm_matrix_text))

    return div+1