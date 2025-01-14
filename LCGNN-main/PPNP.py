import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_sparse import SparseTensor
from torch_geometric.utils import add_self_loops, remove_self_loops, to_dense_adj

class PPNP1(torch.nn.Module):
    def __init__(self, in_channels, out_channels, alpha=0.1, num_layers=1):
        super(PPNP1, self).__init__()
        self.fc = torch.nn.Linear(in_channels, out_channels)
        self.alpha = alpha
        self.num_layers = num_layers

    def forward(self, x, edge_index):
        # 计算归一化的邻接矩阵
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        adj = to_dense_adj(edge_index)[0]  # 获取稠密邻接矩阵
        deg = torch.diag(torch.pow(adj.sum(dim=1), -0.5))
        norm_adj = deg @ adj @ deg

        # 计算 PPNP 聚合
        initial_features = x
        for _ in range(self.num_layers):
            x = self.alpha * initial_features + (1 - self.alpha) * norm_adj @ x
        return x

class PPNP2(torch.nn.Module):
    def __init__(self, in_channels, out_channels, alpha=0.1, num_layers=1):
        super(PPNP2, self).__init__()
        self.fc = torch.nn.Linear(in_channels, out_channels)
        self.alpha = alpha
        self.num_layers = num_layers

    def forward(self, x, edge_index):
        # 计算归一化的邻接矩阵
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        adj = to_dense_adj(edge_index)[0]  # 获取稠密邻接矩阵
        deg = torch.diag(torch.pow(adj.sum(dim=1), -0.5))
        norm_adj = deg @ adj @ deg

        # 计算 PPNP 聚合
        initial_features = x
        for _ in range(self.num_layers):
            x = self.alpha * initial_features + (1 - self.alpha) * norm_adj @ x
        return x
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

# # 初始化 PPNP 模型
# model = PPNP(in_channels=2, out_channels=2, alpha=0.1)
#
# # 使用PPNP对节点特征进行聚合
# out = model(x, edge_index)

# 通过 PPNP 输出补全缺失特征
def ppnp_based_aggregation(original_x, ppnp_output):
    # 找到缺失值（即全为零的向量）
    mask = (original_x == 0).all(dim=1)
    # 将原始特征中的缺失值替换为通过 PPNP 学到的特征值
    original_x[mask] = ppnp_output[mask]
    return original_x

# # 使用PPNP输出对缺失值进行补全
# x_filled = ppnp_based_aggregation(x.clone(), out)
#
# print("补全后的特征矩阵：")
# print(x_filled)
