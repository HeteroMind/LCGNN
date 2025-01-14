import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, to_dense_adj

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

class GCN1(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN1, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x,edge_index):
        # x, edge_index = data.x, data.edge_index
        # 第一层图卷积
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # 第二层图卷积
        x = self.conv2(x, edge_index)
        return x

class GCN2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN2, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x,edge_index):
        # x, edge_index = data.x, data.edge_index
        # 第一层图卷积
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # 第二层图卷积
        x = self.conv2(x, edge_index)
        return x
# # 创建图数据对象
# data = Data(x=x, edge_index=edge_index)

# # 初始化 GCN 模型
# model = GCN(in_channels=2, hidden_channels=4, out_channels=2)

# # 使用模型对节点特征进行聚合
# out = model(data)

# 通过 GCN 输出补全缺失特征
def gcn_based_aggregation(original_x, gcn_output):
    # 找到缺失值（即全为零的向量）
    mask = (original_x == 0).all(dim=1)
    # 将原始特征中的缺失值替换为通过 GCN 学到的特征值
    original_x[mask] = gcn_output[mask]
    return original_x

# # 使用 GCN 输出对缺失值进行补全
# x_filled = gcn_based_aggregation(x.clone(), out)

# print("补全后的特征矩阵：")
# print(x_filled)