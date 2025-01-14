import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops

# # 假设有一个简单的图，包含5个节点和一些边
# edge_index = torch.tensor([[0, 1, 2, 3],
#                            [1, 0, 3, 2]], dtype=torch.long)
#
# # 节点特征矩阵，其中0向量表示缺失值
# x = torch.tensor([[1.0, 2.0],  # 节点0的特征
#                   [2.0, 0.0],  # 节点1的特征（缺失）
#                   [0.0, 3.0],  # 节点2的特征（缺失）
#                   [4.0, 5.0],  # 节点3的特征
#                   [6.0, 7.0]], dtype=torch.float)  # 节点4的特征

# # 创建图数据对象
# data = Data(x=x, edge_index=edge_index)


# Mean Attribute Aggregation
def mean_attribute_aggregation(original_x, edge_index):
    num_nodes = original_x.size(0)
    # 计算邻接矩阵（稀疏）
    row, col = edge_index[0].long(), edge_index[1].long()
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    adj_matrix[row, col] = 1
    adj_matrix[col, row] = 1  # 无向图

    # 计算邻居的均值
    mean_features = torch.zeros_like(original_x)
    for i in range(num_nodes):
        # 识别邻居
        neighbors = adj_matrix[i].nonzero(as_tuple=True)[0]
        # 计算邻居特征均值
        if len(neighbors) > 0:
            neighbor_features = original_x[neighbors]
            mean_features[i] = neighbor_features.mean(dim=0)

    # 用邻居均值替换零特征
    mask = (original_x == 0).all(dim=1)
    original_x[mask] = mean_features[mask]

    return original_x

#
# # 使用 Mean Attribute Aggregation 进行补全
# x_filled = mean_attribute_aggregation(x.clone(), edge_index)

# print("补全后的特征矩阵：")
# print(x_filled)