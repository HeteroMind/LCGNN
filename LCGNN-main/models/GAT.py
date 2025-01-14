import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class GAT(nn.Module):
    def __init__(self,
                 g, #输入的图数据，表示为一个DGL图对象。
                 in_dims,#一个列表，包含了输入图的每种节点特征的维度。
                 num_hidden,#隐藏层的节点特征维度。
                 num_classes,#输出层的节点特征维度，通常对应于任务的类别数。
                 num_layers,# GAT模型的层数。
                 heads,#一个列表，包含每个GAT层的头数（multi-head mechanism）。
                 activation,#GAT层中的激活函数
                 feat_drop,# 特征丢弃率，用于防止过拟合
                 attn_drop,#注意力权重丢弃率，用于防止过拟合
                 negative_slope,#Leaky ReLU激活函数的负斜率。
                 residual, #是否使用残差连接。
                 use_l2norm):  #是否使用L2范数归一化。
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.use_l2norm = use_l2norm
        
        #  fc_list: transform each type of node features into the same dimension num_hidden
        # self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        # for fc in self.fc_list:
        #     nn.init.xavier_normal_(fc.weight, gain=1.414)

        # input projection (no residual)

        # 输入投影，将节点特征投影到num_hidden维度，不使用残差连接
        self.gat_layers.append(GATConv(
            num_hidden, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            # 由于多头机制，输入维度为 num_hidden * num_heads
            # 隐藏层，将输入的节点特征进行多头GAT操作，得到新的节点表示
            self.gat_layers.append(GATConv(
                num_hidden * heads[l - 1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        # 输出投影，将最后一层隐藏层的节点表示投影到num_classes维度
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))
        if self.use_l2norm:
            # 用于进行L2范数归一化的小常数
            self.epsilon = torch.FloatTensor([1e-12]).to(device)#.cuda()

    def forward(self, h): #在前向传播过程中，输入节点特征 h 被传递给每一层的GAT层，最后得到模型的输出。
        for l in range(self.num_layers):
            # 通过每一层的GAT层进行前向传播，将节点表示展平以便传递给下一层
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection  # 输出投影
        logits = self.gat_layers[-1](self.g, h).mean(1)
        if self.use_l2norm:
            # norm    # 如果使用L2范数归一化，进行范数归一化操作
            logits = logits / (torch.max(torch.norm(logits, dim=1, keepdim=True), self.epsilon))
        return h, logits
