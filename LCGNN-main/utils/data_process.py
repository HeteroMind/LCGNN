import torch
import numpy as np
import dgl
import torch.nn as nn

from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def preprocess(features_list, adjM, type_mask, labels, train_val_test_idx, dl, args):#,emb):
    if args.feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif args.feats_type == 6:
        # valid指的是有属性的
        save = args.valid_attributed_type
        feature_dim = features_list[save].shape[1]
        in_dims = []#[features_list[0].shape[1]] + [10] * (len(features_list) - 1)
        for i in range(0, len(features_list)):
            if i == save:
                in_dims.append(feature_dim)
            else:
                in_dims.append(feature_dim)
                # features_list[i] = np.zeros((features_list[i].shape[0], feature_dim))
                features_list[i] = torch.zeros((features_list[i].shape[0], feature_dim)).to(device)
    elif args.feats_type == 1:
        save = args.valid_attributed_type
        feature_dim = features_list[save].shape[0]
        in_dims = []#[features_list[0].shape[1]] + [10] * (len(features_list) - 1)
        for i in range(0, len(features_list)):
            if i == save:
                indices = np.vstack((np.arange(feature_dim), np.arange(feature_dim)))
                indices = torch.LongTensor(indices)
                values = torch.FloatTensor(np.ones(feature_dim))
                features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([feature_dim, feature_dim])).to(device)
                # features_list[i] = np.identity(feature_dim)
                in_dims.append(feature_dim)
            else:
                in_dims.append(feature_dim)
                # features_list[i] = np.zeros((features_list[i].shape[0], feature_dim))
                features_list[i] = torch.zeros((features_list[i].shape[0], feature_dim)).to(device)
    elif args.feats_type == 7:
        save = args.valid_attributed_type
        in_dims = []
        for i in range(0, len(features_list)):
            if i == save:
                feature_dim = features_list[save].shape[1]
                in_dims.append(feature_dim)
            else:
                feature_dim=features_list[i].shape[0]
                indices = np.vstack((np.arange(feature_dim), np.arange(feature_dim)))
                indices = torch.LongTensor(indices)
                values = torch.FloatTensor(np.ones(feature_dim))
                features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([feature_dim, feature_dim])).to(
                    device).to_dense()
                # features_list[i] = np.identity(feature_dim)
                in_dims.append(feature_dim)
    #用于做消融实验
    elif args.feats_type == 8:
        if args.dataset == 'DBLP' and args.missingrate == 0.3:
            #测30%缺失
            features_dim = features_list[1].shape[1]
            in_dims = []
            # 保留0
            features_list[0]=torch.tensor(features_list[0],dtype=torch.float32).to(device)
            original_shape = features_list[0].shape
            if original_shape[-1] < features_dim:
                padding_tensor = torch.zeros(
                    *original_shape[:-1],
                    features_dim - original_shape[-1],
                    dtype=torch.float32,
                    device=features_list[0].device
                )
            features_list[0] = torch.cat((features_list[0],padding_tensor),dim=-1)
            # 保留一
            features_list[1] = torch.tensor(features_list[1],dtype=torch.float32).to(device)

            for i in range(0,len(features_list)):
                if i == 0 or i == 1:
                    in_dims.append(features_dim)
                else:
                    in_dims.append(features_dim)
                    features_list[i] = torch.zeros((features_list[i].shape[0],features_dim)).to(device)
        elif args.dataset == 'DBLP' and args.missingrate == 0.15:
            # 测15%缺失
            features_dim = features_list[1].shape[1] #这里是保留原始的有属性的维度，即需要统一的维度
            in_dims = []
            #保留一
            features_list[1] = torch.tensor(features_list[1], dtype=torch.float32).to(device)
            # 保留二
            features_list[2] = torch.tensor(features_list[2], dtype=torch.float32).to(device)
            original_shape_2 = features_list[2].shape
            if original_shape_2[-1] < features_dim:
                padding_tensor_2 = torch.zeros(
                    *original_shape_2[:-1],
                    features_dim - original_shape_2[-1],
                    dtype=torch.float32,
                    device=features_list[2].device
                )
            features_list[2] = torch.cat((features_list[2], padding_tensor_2), dim=-1)
            # 保留三
            features_list[3] = torch.tensor(features_list[3], dtype=torch.float32).to(device)
            original_shape_3 = features_list[3].shape
            if original_shape_3[-1] < features_dim:
                padding_tensor_3 = torch.zeros(
                    *original_shape_3[:-1],
                    features_dim - original_shape_3[-1],
                    dtype=torch.float32,
                    device=features_list[3].device
                )
            features_list[3] = torch.cat((features_list[3], padding_tensor_3), dim=-1)
            for i in range(0, len(features_list)):
                if i == 1 or i == 2 or i == 3:
                    in_dims.append(features_dim)
                else:
                    in_dims.append(features_dim)
                    features_list[i] = torch.zeros((features_list[i].shape[0], features_dim)).to(device)

        elif args.dataset == 'DBLP' and args.missingrate == 0:
            features_dim = features_list[1].shape[1]  # 这里是保留原始的有属性的维度，即需要统一的维度
            in_dims = [features_dim for _ in range(len(features_list))]
            # 保留一
            features_list[1] = torch.tensor(features_list[1], dtype=torch.float32).to(device)
            # 保留0
            features_list[0] = torch.tensor(features_list[0], dtype=torch.float32).to(device)
            original_shape = features_list[0].shape
            if original_shape[-1] < features_dim:
                padding_tensor = torch.zeros(
                    *original_shape[:-1],
                    features_dim - original_shape[-1],
                    dtype=torch.float32,
                    device=features_list[0].device
                )
            features_list[0] = torch.cat((features_list[0], padding_tensor), dim=-1)
            # 保留二
            features_list[2] = torch.tensor(features_list[2], dtype=torch.float32).to(device)
            original_shape_2 = features_list[2].shape
            if original_shape_2[-1] < features_dim:
                padding_tensor_2 = torch.zeros(
                    *original_shape_2[:-1],
                    features_dim - original_shape_2[-1],
                    dtype=torch.float32,
                    device=features_list[2].device
                )
            features_list[2] = torch.cat((features_list[2], padding_tensor_2), dim=-1)
            # 保留三
            features_list[3] = torch.tensor(features_list[3], dtype=torch.float32).to(device)
            original_shape_3 = features_list[3].shape
            if original_shape_3[-1] < features_dim:
                padding_tensor_3 = torch.zeros(
                    *original_shape_3[:-1],
                    features_dim - original_shape_3[-1],
                    dtype=torch.float32,
                    device=features_list[3].device
                )
            features_list[3] = torch.cat((features_list[3], padding_tensor_3), dim=-1)

        elif args.dataset == 'ACM' and args.missingrate == 0.54:
            # 测54%缺失
            features_dim = features_list[0].shape[1]  # 这里是保留原始的有属性的维度，即需要统一的维度
            in_dims = []
            # 保留0
            features_list[0] = torch.tensor(features_list[0], dtype=torch.float32).to(device)

            # 保留三  但是由于此时0的维度大于三的维度，实行0填充
            features_list[3] = torch.tensor(features_list[3], dtype=torch.float32).to(device)

            for i in range(0, len(features_list)):
                if i == 0 or i == 3:
                    in_dims.append(features_dim)
                else:
                    in_dims.append(features_dim)
                    features_list[i] = torch.zeros((features_list[i].shape[0], features_dim)).to(device)

        elif args.dataset == 'ACM' and args.missingrate == 0.17:
            # 测17%缺失
            features_dim = features_list[0].shape[1]  # 这里是保留原始的有属性的维度，即需要统一的维度
            in_dims = []
            # 保留0
            features_list[0] = torch.tensor(features_list[0], dtype=torch.float32).to(device)
            #保留一
            features_list[1] = torch.tensor(features_list[1], dtype=torch.float32).to(device)

            for i in range(0, len(features_list)):
                if i == 0 or i == 1:
                    in_dims.append(features_dim)
                else:
                    in_dims.append(features_dim)
                    features_list[i] = torch.zeros((features_list[i].shape[0], features_dim)).to(device)

        elif args.dataset == 'ACM' and args.missingrate == 0:
            features_dim = features_list[0].shape[1]  # 这里是保留原始的有属性的维度，即需要统一的维度
            in_dims = [features_dim for _ in range(len(features_list))]
            # 保留0
            features_list[0] = torch.tensor(features_list[0], dtype=torch.float32).to(device)
            # 保留一
            features_list[1] = torch.tensor(features_list[1], dtype=torch.float32).to(device)

            # 保留二
            features_list[2] = torch.tensor(features_list[2], dtype=torch.float32).to(device)

            # 保留三
            features_list[3] = torch.tensor(features_list[3], dtype=torch.float32).to(device)


        elif args.dataset == 'IMDB' and args.missingrate == 0.67:
            # 测67%缺失
            features_dim = features_list[0].shape[1]  # 这里是保留原始的有属性的维度，即需要统一的维度
            in_dims = []
            # 保留0
            features_list[0] = torch.tensor(features_list[0], dtype=torch.float32).to(device)
            # 保留一 但是由于此时0的维度大于一的维度，实行0填充
            features_list[1] = torch.tensor(features_list[1], dtype=torch.float32).to(device)
            original_shape_1 = features_list[1].shape
            if original_shape_1[-1] < features_dim:
                padding_tensor_1 = torch.zeros(
                    *original_shape_1[:-1],
                    features_dim - original_shape_1[-1],
                    dtype=torch.float32,
                    device=features_list[1].device
                )
            features_list[1] = torch.cat((features_list[1], padding_tensor_1), dim=-1)
            for i in range(0, len(features_list)):
                if i == 0 or i == 1:
                    in_dims.append(features_dim)
                else:
                    in_dims.append(features_dim)
                    features_list[i] = torch.zeros((features_list[i].shape[0], features_dim)).to(device)

        elif args.dataset == 'IMDB' and args.missingrate == 0.37:
            # 测37%缺失
            features_dim = features_list[0].shape[1]  # 这里是保留原始的有属性的维度，即需要统一的维度
            in_dims = []
            # 保留0
            features_list[0] = torch.tensor(features_list[0], dtype=torch.float32).to(device)
            # 保留一 但是由于此时0的维度大于一的维度，实行0填充
            features_list[1] = torch.tensor(features_list[1], dtype=torch.float32).to(device)
            original_shape_1 = features_list[1].shape
            if original_shape_1[-1] < features_dim:
                padding_tensor_1 = torch.zeros(
                    *original_shape_1[:-1],
                    features_dim - original_shape_1[-1],
                    dtype=torch.float32,
                    device=features_list[1].device
                )
            features_list[1] = torch.cat((features_list[1], padding_tensor_1), dim=-1)
            # 保留二        但是由于此时0的维度大于一的维度，实行0填充
            features_list[2] = torch.tensor(features_list[2], dtype=torch.float32).to(device)
            original_shape_2 = features_list[2].shape
            if original_shape_2[-1] < features_dim:
                padding_tensor_2 = torch.zeros(
                    *original_shape_2[:-1],
                    features_dim - original_shape_2[-1],
                    dtype=torch.float32,
                    device=features_list[2].device
                )
            features_list[2] = torch.cat((features_list[2], padding_tensor_2), dim=-1)
            for i in range(0, len(features_list)):
                if i == 0 or i == 1 or i == 2:
                    in_dims.append(features_dim)
                else:
                    in_dims.append(features_dim)
                    features_list[i] = torch.zeros((features_list[i].shape[0], features_dim)).to(device)

        elif args.dataset == 'IMDB' and args.missingrate == 0:
            # 测37%缺失
            features_dim = features_list[0].shape[1]  # 这里是保留原始的有属性的维度，即需要统一的维度
            in_dims = [features_dim for _ in range(len(features_list))]
            # 保留0
            features_list[0] = torch.tensor(features_list[0], dtype=torch.float32).to(device)
            # 保留一 但是由于此时0的维度大于一的维度，实行0填充
            features_list[1] = torch.tensor(features_list[1], dtype=torch.float32).to(device)
            original_shape_1 = features_list[1].shape
            if original_shape_1[-1] < features_dim:
                padding_tensor_1 = torch.zeros(
                    *original_shape_1[:-1],
                    features_dim - original_shape_1[-1],
                    dtype=torch.float32,
                    device=features_list[1].device
                )
            features_list[1] = torch.cat((features_list[1], padding_tensor_1), dim=-1)
            # 保留二        但是由于此时0的维度大于一的维度，实行0填充
            features_list[2] = torch.tensor(features_list[2], dtype=torch.float32).to(device)
            original_shape_2 = features_list[2].shape
            if original_shape_2[-1] < features_dim:
                padding_tensor_2 = torch.zeros(
                    *original_shape_2[:-1],
                    features_dim - original_shape_2[-1],
                    dtype=torch.float32,
                    device=features_list[2].device
                )
            features_list[2] = torch.cat((features_list[2], padding_tensor_2), dim=-1)
            # 保留三
            features_list[3] = torch.tensor(features_list[3], dtype=torch.float32).to(device)
            features_list[3] = features_list[3][:, :features_dim]




















    # in_dims = [features.shape[1]  for features in features_list]#这是使用用DP_AC_2。py的

    #in_dims = [features_list.shape[1]] * args.max_num_views#这是使用用DP_AC。py的
    # labels = torch.LongTensor(labels).to(device)
    train_idx = train_val_test_idx['train_idx']
    train_idx = np.sort(train_idx)
    val_idx = train_val_test_idx['val_idx']
    val_idx = np.sort(val_idx)
    test_idx = train_val_test_idx['test_idx']
    test_idx = np.sort(test_idx)

    if args.dataset == 'IMDB':
        g = dgl.DGLGraph(adjM)
    else:
        g = dgl.DGLGraph(adjM + (adjM.T))
    g = dgl.remove_self_loop(g)#这行代码的作用是移除图 g 中的自环（self-loop）。自环是指连接节点到其自身的边。该函数会检查图中是否存在自环，并将自环从图中移除。这可以确保在图神经网络中进行训练和处理时，不考虑节点与其自身之间的连接。
    g = dgl.add_self_loop(g)#这行代码的作用是将自环添加回图 g。即使在第一步中移除了自环，在某些情况下，特别是在图卷积神经网络 (GCN) 等模型中，可能需要在训练前将自环重新添加到图中。自环可以使节点保留其自身特征并参与信息传播。
    g = g.to(device)#这行代码将图 g 移动到指定的 device 上进行计算。这是将图数据移动到 GPU 或其他特定设备进行加速计算的常见操作。device 可能是指定的硬件设备，比如 'cuda:0' 表示使用 GPU 0 进行计算。
    num_classes = dl.labels_train['num_classes']


    #原autoac代码中只含有这部分的损失函数的代码段
    # if args.dataset == 'IMDB':
    #     criterion = nn.BCELoss()#nn.BCELoss() 是一个用于二分类问题的损失函数，通常用于计算二进制交叉熵损失。
    # else:
    #     criterion = nn.CrossEntropyLoss()#是一个用于多分类问题的损失函数。通常用于多分类任务中，尤其是当目标类别具有两个或更多类别时。在神经网络中，交叉熵损失函数用于衡量模型输出的概率分布与实际标签的差异。对于多分类任务，输出通常是一个概率分布，而 CrossEntropyLoss 会将这个输出与实际标签进行比较，计算出预测结果与实际标签之间的交叉熵损失，然后将这个损失作为模型优化的依据，通过反向传播算法来调整网络的权重。在给定的代码中，criterion = nn.CrossEntropyLoss() 表示使用了交叉熵损失函数作为模型的损失函数。这个选择适用于多分类任务，可以用于训练神经网络，尤其是当目标类别超过两个类别时。
    #
    # criterion = criterion.cuda()
    # criterion = nn.BCEWithLogitsLoss()  # 二元交叉熵损失函数.通常用于链路预测任务中
    #节点分类实验的损失函数
    if args.dataset == 'IMDB':
        criterion = nn.BCELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # 只在需要时将 criterion 移动到 GPU 上
    if torch.cuda.is_available():
        criterion = criterion.cuda()
    #链路预测实验的损失函数
    # class MarginLoss(nn.Module):
    #     def __init__(self, margin):
    #         super(MarginLoss, self).__init__()
    #         self.margin = margin
    #
    #     def forward(self, output1, output2, target):
    #         distance = torch.norm(output1 - output2, p=2, dim=1)  # 计算两个向量之间的欧氏距离
    #         loss = torch.mean(torch.max(0, self.margin - distance * target) ** 2)  # 计算Margin损失
    #         return loss
    # if args.dataset == 'IMDB':
    #     criterion = nn.BCEWithLogitsLoss()  # 二元交叉熵损失函数，适用于链路预测任务
    # else:
    #     # 可根据具体情况选择其他损失函数，如Margin损失函数等
    #     criterion = MarginLoss(margin=0.5)
    #
    #     # 只在需要时将 criterion 移动到 GPU 上
    # if torch.cuda.is_available():
    #     criterion = criterion.cuda()
    # model = GCN(features_list[args.valid_attributed_type].shape(1), args.hidden_dim, args.max_features_len)
    # # 创建 PyTorch Geometric 数据对象
    # data = Data(x=features_list, edge_index=g.edata)#edge_index=edge_index_with_self_loops)
    # # 前向传播以获得降维后的节点特征
    # features_list = model(data.x, data.edge_index)
    return (features_list, labels, g, type_mask, dl, in_dims, num_classes), (train_idx, val_idx, test_idx), criterion



'''
这段代码是一个数据预处理的函数，根据给定的异质图数据进行预处理。主要的功能包括：

根据指定的特征类型（args.feats_type），确定输入特征的维度（in_dims）。
根据输入的特征列表、邻接矩阵、节点类型、标签等信息，生成 DGL 图对象（g）。
根据数据集类型（args.dataset），选择相应的损失函数（criterion）。
准备训练集、验证集和测试集的索引。
此函数将异质图数据准备成模型可用的形式，并返回一个包含模型训练所需数据的元组。

需要注意的是，这段代码中涉及了设备的选择（GPU 或 CPU），根据可用的 CUDA 设备自动选择计算设备，并将相关数据转移到该设备上进行处理。同时，根据不同数据集选择了不同的损失函数进行训练。'''
'''
这段代码是一个用于数据预处理的函数，其目的是将异质图数据（包括特征、邻接矩阵、标签等）进行准备，以便用于模型训练。下面是函数的主要步骤和处理过程：

特征处理：
根据参数 args.feats_type 的不同取值，对输入的特征列表 features_list 进行处理以确定输入特征的维度 in_dims。

当 args.feats_type == 0 时，特征的维度由特征列表中各个特征的形状的第二个维度确定。
当 args.feats_type == 6 或 args.feats_type == 1 时，根据 args.valid_attributed_type 的值选择特定类型的特征，其维度也由这个特定类型的特征的形状确定。
标签和索引处理：

将标签数据 labels 转换为 PyTorch Tensor，并将其移到 GPU 上（如果可用）。
获取训练集、验证集和测试集的索引并进行排序。
创建图对象：

根据异质图的邻接矩阵 adjM 构建 DGL 图对象 g。
对图进行去除自环（self-loop）和添加自环的操作，并将图对象移动到 GPU 上（如果可用）。
选择损失函数：

根据数据集类型 args.dataset 的不同，选择不同的损失函数。如果是 IMDB 数据集，选用 nn.BCELoss()（用于二分类问题）；其他数据集选用 nn.CrossEntropyLoss()（用于多分类问题）。
其他设置：

获取数据加载器中的类别数 num_classes。
返回数据：

将处理后的特征列表、标签、图对象、维度信息、类别数、训练集、验证集和测试集索引以及所选择的损失函数作为元组返回。
总的来说，该函数通过对输入数据进行预处理，创建了适用于模型训练的数据集，并进行了一些设备选择和数据处理的操作，使数据能够在 PyTorch 框架中进行有效地使用和训练。'''
