import os
import numpy as np
import scipy.sparse as sp
from collections import Counter, defaultdict
from sklearn.metrics import f1_score
import time
from sklearn.metrics import roc_auc_score
#from data import *
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class data_loader:
    def __init__(self, path):
        self.path = path
        self.nodes = self.load_nodes()#这里跳到本.py文件的第305行
        self.links = self.load_links()
        self.labels_train = self.load_labels('label.dat')
        self.labels_test = self.load_labels('label.dat.test')

    def get_sub_graph(self, node_types_tokeep):
        """
        node_types_tokeep is a list or set of node types that you want to keep in the sub-graph
        We only support whole type sub-graph for now.
        This is an in-place update function!
        return: old node type id to new node type id dict, old edge type id to new edge type id dict
        """
        keep = set(node_types_tokeep)
        new_node_type = 0
        new_node_id = 0
        new_nodes = {'total':0, 'count':Counter(), 'attr':{}, 'shift':{}}
        new_links = {'total':0, 'count':Counter(), 'meta':{}, 'data':defaultdict(list)}
        new_labels_train = {'num_classes':0, 'total':0, 'count':Counter(), 'data':None, 'mask':None}
        new_labels_test = {'num_classes':0, 'total':0, 'count':Counter(), 'data':None, 'mask':None}
        old_nt2new_nt = {}
        old_idx = []
        for node_type in self.nodes['count']:
            if node_type in keep:
                nt = node_type
                nnt = new_node_type
                old_nt2new_nt[nt] = nnt
                cnt = self.nodes['count'][nt]
                new_nodes['total'] += cnt
                new_nodes['count'][nnt] = cnt
                new_nodes['attr'][nnt] = self.nodes['attr'][nt]
                new_nodes['shift'][nnt] = new_node_id
                beg = self.nodes['shift'][nt]
                old_idx.extend(range(beg, beg+cnt))
                
                cnt_label_train = self.labels_train['count'][nt]
                new_labels_train['count'][nnt] = cnt_label_train
                new_labels_train['total'] += cnt_label_train
                cnt_label_test = self.labels_test['count'][nt]
                new_labels_test['count'][nnt] = cnt_label_test
                new_labels_test['total'] += cnt_label_test
                
                new_node_type += 1
                new_node_id += cnt

        new_labels_train['num_classes'] = self.labels_train['num_classes']
        new_labels_test['num_classes'] = self.labels_test['num_classes']
        for k in ['data', 'mask']:
            new_labels_train[k] = self.labels_train[k][old_idx]
            new_labels_test[k] = self.labels_test[k][old_idx]

        old_et2new_et = {}
        new_edge_type = 0
        for edge_type in self.links['count']:
            h, t = self.links['meta'][edge_type]
            if h in keep and t in keep:
                et = edge_type
                net = new_edge_type
                old_et2new_et[et] = net
                new_links['total'] += self.links['count'][et]
                new_links['count'][net] = self.links['count'][et]
                new_links['meta'][net] = tuple(map(lambda x:old_nt2new_nt[x], self.links['meta'][et]))
                new_links['data'][net] = self.links['data'][et][old_idx][:, old_idx]
                new_edge_type += 1

        self.nodes = new_nodes
        self.links = new_links
        self.labels_train = new_labels_train
        self.labels_test = new_labels_test
        return old_nt2new_nt, old_et2new_et

    def get_meta_path(self, meta=[]):
        """
        Get meta path matrix
            meta is a list of edge types (also can be denoted by a pair of node types)
            return a sparse matrix with shape [node_num, node_num]
        """
        ini = sp.eye(self.nodes['total'])
        meta = [self.get_edge_type(x) for x in meta]
        for x in meta:
            ini = ini.dot(self.links['data'][x]) if x >= 0 else ini.dot(self.links['data'][-x - 1].T)
        return ini

    def dfs(self, now, meta, meta_dict):
        if len(meta) == 0:
            meta_dict[now[0]].append(now)
            return
        th_mat = self.links['data'][meta[0]] if meta[0] >= 0 else self.links['data'][-meta[0] - 1].T
        th_node = now[-1]
        for col in th_mat[th_node].nonzero()[1]:
            self.dfs(now+[col], meta[1:], meta_dict)

    def get_full_meta_path(self, meta=[], symmetric=False):
        """
        Get full meta path for each node
            meta is a list of edge types (also can be denoted by a pair of node types)
            return a dict of list[list] (key is node_id)
        """
        meta = [self.get_edge_type(x) for x in meta]
        if len(meta) == 1:
            meta_dict = {}
            start_node_type = self.links['meta'][meta[0]][0] if meta[0]>=0 else self.links['meta'][-meta[0]-1][1]
            for i in range(self.nodes['shift'][start_node_type], self.nodes['shift'][start_node_type]+self.nodes['count'][start_node_type]):
                meta_dict[i] = []
                self.dfs([i], meta, meta_dict)
        else:
            meta_dict1 = {}
            meta_dict2 = {}
            mid = len(meta) // 2
            meta1 = meta[:mid]
            meta2 = meta[mid:]
            start_node_type = self.links['meta'][meta1[0]][0] if meta1[0]>=0 else self.links['meta'][-meta1[0]-1][1]
            for i in range(self.nodes['shift'][start_node_type], self.nodes['shift'][start_node_type]+self.nodes['count'][start_node_type]):
                meta_dict1[i] = []
                self.dfs([i], meta1, meta_dict1)
            start_node_type = self.links['meta'][meta2[0]][0] if meta2[0]>=0 else self.links['meta'][-meta2[0]-1][1]
            for i in range(self.nodes['shift'][start_node_type], self.nodes['shift'][start_node_type]+self.nodes['count'][start_node_type]):
                meta_dict2[i] = []
            if symmetric:
                for k in meta_dict1:
                    paths = meta_dict1[k]
                    for x in paths:
                        meta_dict2[x[-1]].append(list(reversed(x)))
            else:
                for i in range(self.nodes['shift'][start_node_type], self.nodes['shift'][start_node_type]+self.nodes['count'][start_node_type]):
                    self.dfs([i], meta2, meta_dict2)
            meta_dict = {}
            start_node_type = self.links['meta'][meta1[0]][0] if meta1[0]>=0 else self.links['meta'][-meta1[0]-1][1]
            for i in range(self.nodes['shift'][start_node_type], self.nodes['shift'][start_node_type]+self.nodes['count'][start_node_type]):
                meta_dict[i] = []
                for beg in meta_dict1[i]:
                    for end in meta_dict2[beg[-1]]:
                        meta_dict[i].append(beg + end[1:])
        return meta_dict

    def gen_file_for_evaluate(self, test_idx, label, file_path, mode='bi'):
        print(test_idx.shape, label.shape)
        if test_idx.shape[0] != label.shape[0]:
            return
        print(os.path.abspath(file_path))
        # print(f"123 {mode}")
        if mode == 'multi':
            print(mode)
            multi_label = []
            for i in range(label.shape[0]):
                label_list = [str(j) for j in range(label[i].shape[0]) if label[i][j]==1]
                multi_label.append(','.join(label_list))
            label = multi_label
        elif mode=='bi':
            label = np.array(label)
        else:
            return
        with open(file_path, "w") as f:
            for nid, l in zip(test_idx, label):
                f.write(f"{nid}\t\t{self.get_node_type(nid)}\t{l}\n")

    def evaluate(self, pred):
        print(f"{bcolors.WARNING}Warning: If you want to obtain test score, please submit online on biendata.{bcolors.ENDC}")
        y_true = self.labels_test['data'][self.labels_test['mask']]
        micro = f1_score(y_true, pred, average='micro')
        macro = f1_score(y_true, pred, average='macro')
        #roc_auc = roc_auc_score(y_true, pred)

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
        # mrr = mean_reciprocal_rank(y_true, pred)
        #
        result = {
            'micro-f1': micro,
            'macro-f1': macro,
        }
        # result = {
        #     'micro-f1': micro,
        #     'macro-f1': macro,
        #     'roc_auc': roc_auc,
        #     'MRR score': mrr
        # }
        return result

    def load_labels(self, name):
        """
        return labels dict
            num_classes: total number of labels
            total: total number of labeled data
            count: number of labeled data for each node type
            data: a numpy matrix with shape (self.nodes['total'], self.labels['num_classes'])
            mask: to indicate if that node is labeled, if False, that line of data is masked
        """
        labels = {'num_classes':0, 'total':0, 'count':Counter(), 'data':None, 'mask':None}
        nc = 0
        mask = np.zeros(self.nodes['total'], dtype=bool)
        data = [None for i in range(self.nodes['total'])]
        with open(os.path.join(self.path, name), 'r', encoding='utf-8') as f:
            for line in f:
                th = line.split('\t')
                node_id, node_name, node_type, node_label = int(th[0]), th[1], int(th[2]), list(map(int, th[3].split(',')))
                for label in node_label:
                    nc = max(nc, label+1)
                mask[node_id] = True
                data[node_id] = node_label
                labels['count'][node_type] += 1
                labels['total'] += 1
        labels['num_classes'] = nc
        new_data = np.zeros((self.nodes['total'], labels['num_classes']), dtype=int)
        for i,x in enumerate(data):
            if x is not None:
                for j in x:
                    new_data[i, j] = 1
        labels['data'] = new_data
        labels['mask'] = mask
        return labels

    def get_node_type(self, node_id):
        for i in range(len(self.nodes['shift'])):
            if node_id < self.nodes['shift'][i]+self.nodes['count'][i]:
                return i

    def get_edge_type(self, info):
        if type(info) is int or len(info) == 1:
            return info
        for i in range(len(self.links['meta'])):
            if self.links['meta'][i] == info:
                return i
        info = (info[1], info[0])
        for i in range(len(self.links['meta'])):
            if self.links['meta'][i] == info:
                return -i - 1
        raise Exception('No available edge type')

    def get_edge_info(self, edge_id):
        return self.links['meta'][edge_id]
    
    def list_to_sp_mat(self, li):
        data = [x[2] for x in li]
        i = [x[0] for x in li]
        j = [x[1] for x in li]
        return sp.coo_matrix((data, (i,j)), shape=(self.nodes['total'], self.nodes['total'])).tocsr()
    '''这里加载我想要的边集！！！！！！！！！'''
    def load_links(self):
        """
        return links dict
            total: total number of links
            count: a dict of int, number of links for each type
            meta: a dict of tuple, explaining the link type is from what type of node to what type of node
            data: a dict of sparse matrices, each link type with one matrix. Shapes are all (nodes['total'], nodes['total'])
        """
        '''
        这段代码的作用是读取一个名为 'link.dat' 的文件，并解析其中的内容，将其组织成一个名为 links 的数据结构。代码通过处理文件中的每一行，将其分割成元素列表，然后将列表中的元素提取为整数或浮点数，存储在 links 结构中的不同部分。

        具体地：
        
        links 是一个字典，包含了不同的键，每个键关联到不同的值。
        'total' 键对应的值用于存储总的链接数。
        'count' 键对应的值是一个 Counter 对象，用于记录不同关系类型的链接数量。
        'meta' 键对应的值是一个字典，用于存储关系ID到元组（h_type, t_type）的映射关系。
        'data' 键对应的值是一个 defaultdict，用于存储关系ID到包含元组 (h_id, t_id, link_weight) 的列表。
        代码遍历文件中的每一行，并使用 line.split('\t') 方法按制表符分割每行的内容，将其拆分为 th 列表。然后，从 th 列表中提取元素，将其转换为整数或浮点数类型，并将结果存储在相应的数据结构中。
        
        在处理文件中的每一行时：
        
        h_id, t_id, r_id, link_weight 分别是从拆分后的 th 列表中获取的前四个元素。
        如果 r_id 不在 links['meta'] 中，则会获取 h_id 和 t_id 的节点类型，并将它们存储在 links['meta'] 中。
        无论是否出现重复的 r_id，都会将 (h_id, t_id, link_weight) 添加到 links['data'][r_id] 中。
        links['count'][r_id] 记录了每个关系类型的链接数量。
        links['total'] 统计了总的链接数量。
        这段代码的主要目的是解析文件中的行，提取信息并将其存储在合适的数据结构中，以便后续的分析和处理。'''
        #zzzzzzzzzzzzzz这里很重要！！！！！！！
        links = {'total':0, 'count':Counter(), 'meta':{}, 'data':defaultdict(list)}
        with open(os.path.join(self.path, 'link.dat'), 'r', encoding='utf-8') as f:
            for line in f:
                th = line.split('\t')
                h_id, t_id, r_id, link_weight = int(th[0]), int(th[1]), int(th[2]), float(th[3])
                if r_id not in links['meta']:
                    h_type = self.get_node_type(h_id)#284行跳到224行进行获取节点类型操作
                    t_type = self.get_node_type(t_id)
                    links['meta'][r_id] = (h_type, t_type)
                links['data'][r_id].append((h_id, t_id, link_weight))
                links['count'][r_id] += 1
                links['total'] += 1
        new_data = {}
        for r_id in links['data']:
            new_data[r_id] = self.list_to_sp_mat(links['data'][r_id])
        links['data'] = new_data
        return links
    #在这里加载了节点数据集后，进行上方的边数据集加载
    def load_nodes(self):
        """
        return nodes dict
            total: total number of nodes
            count: a dict of int, number of nodes for each type
            attr: a dict of np.array (or None), attribute matrices for each type of nodes
            shift: node_id shift for each type. You can get the id range of a type by 
                        [ shift[node_type], shift[node_type]+count[node_type] )
        """
        print(os.path.abspath(self.path))
        # print(os.path.abspath('../../data/IMDB'))
        '''
        #这里进行的是节点的搜寻！！！或许这里是我想要的！！！！！！！！！！！！！！289行是到310行是直接加载了所有的节点，例如：DBLP数据集中，其中节点author（4057,334），paper节点（14328,4231），term节点（7723,50），venue节点（20,20）
        '''
        nodes = {'total':0, 'count':Counter(), 'attr':{}, 'shift':{}, 'shift_end': {}}
        with open(os.path.join(self.path, 'node.dat'), 'r', encoding='utf-8') as f:
            for line in f:
                th = line.split('\t')
                if len(th) == 4:
                    # Then this line of node has attribute
                    node_id, node_name, node_type, node_attr = th#这里获取的就是数据集中的节点id，节点名字、节点类型、节点属性
                    node_id = int(node_id)
                    node_type = int(node_type)
                    node_attr = list(map(float, node_attr.split(',')))
                    nodes['count'][node_type] += 1
                    nodes['attr'][node_id] = node_attr
                    nodes['total'] += 1
                elif len(th) == 3:
                    # Then this line of node doesn't have attribute
                    node_id, node_name, node_type = th
                    node_id = int(node_id)
                    node_type = int(node_type)
                    nodes['count'][node_type] += 1
                    nodes['total'] += 1
                else:
                    raise Exception("Too few information to parse!")
        shift = 0
        attr = {}
        for i in range(len(nodes['count'])):
            nodes['shift'][i] = shift
            if shift in nodes['attr']:
                mat = []
                for j in range(shift, shift+nodes['count'][i]):
                    mat.append(nodes['attr'][j])
                attr[i] = np.array(mat)
            else:
                attr[i] = None
            shift += nodes['count'][i]
            nodes['shift_end'][i] = shift - 1
        nodes['attr'] = attr
        return nodes  #这里搜寻完之后跳到data_loader.py文件的

# if __name__ == '__main__':
#     prefix = 'IMDB'
#     dirs = 'data/'
#     print(os.path.abspath(dirs))
#     if os.path.exists(dirs):
#         print('yes')

'''
这段代码定义了一个名为 data_loader 的类，其中包含了加载异质图数据和对数据进行处理的一些方法。以下是该类中各个方法的主要功能：

__init__: 初始化函数，根据给定路径加载数据集的节点、链接和标签信息。

get_sub_graph: 获取子图，根据指定节点类型保留对应的节点和链接信息，返回旧节点类型到新节点类型的映射。

get_meta_path: 根据给定的元路径（即一系列边的类型），获取元路径的邻接矩阵。

dfs: 深度优先搜索函数，用于生成完整的元路径。

get_full_meta_path: 获取完整的元路径，包括起始节点和结束节点之间所有可能的路径。

gen_file_for_evaluate: 生成用于评估的文件。

evaluate: 评估预测结果的性能。

load_labels, load_links, load_nodes: 加载标签、链接和节点数据。

该类提供了加载数据、生成元路径等功能，可以方便地处理异质图数据，并为后续的分析和评估提供基础支持。'''