import dgl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class HGTLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, dropout = 0.2, use_norm = False):
        super(HGTLayer, self).__init__()

        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.num_types     = num_types
        self.num_relations = num_relations
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        
        self.k_linears   = nn.ModuleList()
        self.q_linears   = nn.ModuleList()
        self.v_linears   = nn.ModuleList()
        self.a_linears   = nn.ModuleList()
        self.norms       = nn.ModuleList()
        self.use_norm    = use_norm
        
        for t in range(num_types):
            self.k_linears.append(nn.Linear(in_dim,   out_dim))
            self.q_linears.append(nn.Linear(in_dim,   out_dim))
            self.v_linears.append(nn.Linear(in_dim,   out_dim))
            self.a_linears.append(nn.Linear(out_dim,  out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))
            
        self.relation_pri   = nn.Parameter(torch.ones(num_relations, self.n_heads))
        self.relation_att   = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg   = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.skip           = nn.Parameter(torch.ones(num_types))
        self.drop           = nn.Dropout(dropout)
        
        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)
    '''
    下段代码定义了一个名为 edge_attention 的函数，用于在图的边上执行操作。让我们逐行解释这个函数：

    etype = edges.data['id'][0]: 该行代码获取了边的类型（即边的标识符），这个类型信息可能存储在边数据的 'id' 字段中。
    
    relation_att = self.relation_att[etype]: 获取了特定边类型 etype 对应的关系属性 relation_att。
    
    relation_pri = self.relation_pri[etype]: 获取了特定边类型 etype 对应的关系优先级 relation_pri。
    
    relation_msg = self.relation_msg[etype]: 获取了特定边类型 etype 对应的关系消息 relation_msg。
    
    key = torch.bmm(edges.src['k'].transpose(1, 0), relation_att).transpose(1, 0): 这行代码计算了边源节点的关键信息 key。在这里，源节点的 'k' 属性与 relation_att 之间进行了矩阵乘法操作。
    
    att = (edges.dst['q'] * key).sum(dim=-1) * relation_pri / self.sqrt_dk: 该行代码计算了边的注意力权重 att。这里使用了目标节点的 'q' 属性与计算出的 key 的乘积，通过求和并应用关系优先级和其它参数进行归一化。
    
    val = torch.bmm(edges.src['v'].transpose(1, 0), relation_msg).transpose(1, 0): 这行代码计算了边源节点的值 val。它使用了源节点的 'v' 属性与 relation_msg 之间的矩阵乘法操作。
    
    最后，该函数返回了包含计算出的注意力权重 att 和值 val 的字典：{'a': att, 'v': val}。
    
    总体来说，这个函数的作用是针对图的特定边类型，计算边的注意力权重和对应的值，以字典的形式返回这些计算结果'''
    def edge_attention(self, edges):
        etype = edges.data['id'][0]
        relation_att = self.relation_att[etype]
        relation_pri = self.relation_pri[etype]
        relation_msg = self.relation_msg[etype]
        key   = torch.bmm(edges.src['k'].transpose(1,0), relation_att).transpose(1,0)
        att   = (edges.dst['q'] * key).sum(dim=-1) * relation_pri / self.sqrt_dk
        val   = torch.bmm(edges.src['v'].transpose(1,0), relation_msg).transpose(1,0)
        return {'a': att, 'v': val}
    
    def message_func(self, edges):
        return {'v': edges.data['v'], 'a': edges.data['a']}
    
    def reduce_func(self, nodes):
        att = F.softmax(nodes.mailbox['a'], dim=1)
        h   = torch.sum(att.unsqueeze(dim = -1) * nodes.mailbox['v'], dim=1)
        return {'t': h.view(-1, self.out_dim)}
    '''
    下段代码看起来是一个神经网络模型中的前向传播函数。它似乎是用于处理图神经网络中的消息传递或注意力机制。在这个函数中：

    self.k_linears, self.v_linears, self.q_linears 似乎是一些线性层（或线性变换），它们通过 node_dict 中的节点类型索引来选择相应的线性层。
    G.canonical_etypes 返回图中规范化的边类型（源节点类型、边类型、目标节点类型）的元组列表。
    循环遍历了 G.canonical_etypes 中的每个边类型，并根据源节点类型和目标节点类型来选择相应的线性层。
    该代码段的目的可能是对不同节点类型之间的信息传递或者特定边的特征提取进行处理。可能在循环内部的代码中，根据边的不同类型执行了不同的信息聚合、特征转换或者注意力机制的操作。
    其中：在给定的 canonical_etypes 中：
    
    srctype 代表边的源节点类型。在 canonical_etypes 中，srctype 的可能取值是：'0', '1', '1', '1', '2', '3'。
    etype 代表边的类型。在 canonical_etypes 中，etype 的可能取值是：'0_1', '1_0', '1_2', '1_3', '2_1', '3_1'。
    dsttype 代表边的目标节点类型。在 canonical_etypes 中，dsttype 的可能取值是：'1', '0', '2', '3', '1', '1'。
    因此，在执行 for srctype, etype, dsttype in G.canonical_etypes: 后：
    
    第一次迭代中，srctype 的值为 '0'，etype 的值为 '0_1'，dsttype 的值为 '1'。
    第二次迭代中，srctype 的值为 '1'，etype 的值为 '1_0'，dsttype 的值为 '0'。
    以此类推，依次进行迭代。'''
    def forward(self, G, inp_key, out_key):
        node_dict, edge_dict = G.node_dict, G.edge_dict
        for srctype, etype, dsttype in G.canonical_etypes:#G.canonical_etypes的值是[('0', '0_1', '1'), ('1', '1_0', '0'), ('1', '1_2', '2'), ('1', '1_3', '3'), ('2', '2_1', '1'), ('3', '3_1', '1')]，# 因此对应的元组进行for循环，对应位置
            #!!!!!!!!!!!!!!!!!
            #print(srctype,etype,dsttype)
            #!!!!!!!!!!!!!!!!!
            k_linear = self.k_linears[node_dict[srctype]]
            v_linear = self.v_linears[node_dict[srctype]] 
            q_linear = self.q_linears[node_dict[dsttype]]
            
            G.nodes[srctype].data['k'] = k_linear(G.nodes[srctype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            G.nodes[srctype].data['v'] = v_linear(G.nodes[srctype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            G.nodes[dsttype].data['q'] = q_linear(G.nodes[dsttype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            
            G.apply_edges(func=self.edge_attention, etype=etype)
        G.multi_update_all({etype : (self.message_func, self.reduce_func) \
                            for etype in edge_dict}, cross_reducer = 'mean')
        for ntype in G.ntypes:
            n_id = node_dict[ntype]
            alpha = torch.sigmoid(self.skip[n_id])
            trans_out = self.a_linears[n_id](G.nodes[ntype].data['t'])
            trans_out = trans_out * alpha + G.nodes[ntype].data[inp_key] * (1-alpha)
            if self.use_norm:
                G.nodes[ntype].data[out_key] = self.drop(self.norms[n_id](trans_out))
            else:
                G.nodes[ntype].data[out_key] = self.drop(trans_out)
    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_types={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)
                
class HGT(nn.Module):
    def __init__(self, G, n_inps, n_hid, n_out, n_layers, n_heads, use_norm = True):
        super(HGT, self).__init__()
        self.gcs = nn.ModuleList()
        self.n_inps = n_inps
        self.n_hid = n_hid
        self.n_out = n_out
        self.n_layers = n_layers
        self.adapt_ws  = nn.ModuleList()
        # 已经做过映射了
        for t in range(len(G.node_dict)):
            self.adapt_ws.append(nn.Linear(n_inps[t], n_hid))
        for _ in range(n_layers):
            self.gcs.append(HGTLayer(n_hid, n_hid, len(G.node_dict), len(G.edge_dict), n_heads, use_norm = use_norm))
        self.out = nn.Linear(n_hid, n_out)

    def forward(self, G, out_key, features_list):
        for ntype in G.ntypes:
            G.nodes[ntype].data['inp'] = features_list[int(ntype)]#.to(device)
        # for ntype in G.ntypes:
        #     n_id = G.node_dict[ntype]
        #     G.nodes[ntype].data['h'] = torch.tanh(self.adapt_ws[n_id](G.nodes[ntype].data['inp']))
        for ntype in G.ntypes:
            n_id = G.node_dict[ntype]
            G.nodes[ntype].data['h'] = torch.tanh(G.nodes[ntype].data['inp'])
        for i in range(self.n_layers):
            self.gcs[i](G, 'h', 'h')
        node_embedding_outkey = G.nodes[out_key].data['h']
        node_embedding = []
        for ntype in G.ntypes:
            node_embedding.append(G.nodes[ntype].data['h'])
        node_embedding = torch.cat(node_embedding, 0)
        return node_embedding, self.out(node_embedding_outkey)
    
    def __repr__(self):
        return '{}(n_inp={}, n_hid={}, n_out={}, n_layers={})'.format(
            self.__class__.__name__, self.n_inp, self.n_hid,
            self.n_out, self.n_layers)
