import torch.nn.functional as F
from torch import nn
import torch
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv

from typing import Any, Callable, List, NamedTuple, Optional
from functools import partial

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=128):#64
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

        self.project1 = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        # # first channel
        w = self.project(z)
        beta = torch.softmax(w, dim=0)
        x1 = (beta * z).sum(0)
        # second channel
        w1 = self.project1(z)
        beta1 = torch.softmax(w1, dim=0)
        x2 = (beta1 * z).sum(0)

        # 双通道注意力聚合！！
        attention_weights = F.softmax(torch.randn(x1.size(0), 1), dim=0).to(device)
        x = attention_weights * x1 + (1 - attention_weights) * x2
        # #动态权重生成
        #
        # attention_score = self.dunamic_weight_net(z)
        # dunamic_weight=F.softmax(attention_score,dim=2)
        #
        #
        # x = (dunamic_weight[:,:,0:1] * x1 + dunamic_weight[:,:,1:2] * x2)
        # if attention_score[:,:,0].mean() > attention_score[:,:,1].mean():
        #     x = x[0]
        # else:
        #     x = x[1]

        return x, beta


class M_GCN_t_noHGNN(nn.Module):
    def __init__(self, num_features_list, hidden_dim,args,dl,data_info):
        super().__init__()
        self.view_num = len(num_features_list)
        self.conv1 = nn.ModuleList([GCNConv(in_channels, hidden_dim) for in_channels in num_features_list])  #GCN 92.59
        self.conv2 = nn.ModuleList([GATConv(hidden_dim, hidden_dim) for _ in range(self.view_num)])#GCNConv  #GAT 92.59
        #self.multihead_attn = nn.MultiheadAttention(hidden_dim, num_heads=4)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4, dropout=0.1) #heads 8     *4     0.2   92.59
        self.encoder_layer1 = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4, dropout=0.1) #heads  8     *4    0.2  92.59
        #self.encoder_layer = EncoderBlock(hidden_dim=hidden_dim, num_heads=4, dropout=0.1, attention_dropout=0.1, mlp_dim=hidden_dim * 4)
        #self.encoder_layer1 = EncoderBlock(hidden_dim=hidden_dim, num_heads=4, dropout=0.1, attention_dropout=0.1,mlp_dim=hidden_dim * 4)
        self.dropout = nn.Dropout(0.3) #0.2   92.59
        self.layer_norm = torch.nn.LayerNorm(hidden_dim)
        self.attention = Attention(hidden_dim)

        self.args = args
        self.dl=dl
        self.g = data_info[2]

        self.myGATconv=myGATConv(self.args.attn_vec_dim, self.dl.labels_train['num_classes'], 1,
            self.args.dropout, self.args.dropout, self.args.slope, True, None, alpha=0.00).to(device)
        self.epsilon = torch.FloatTensor([1e-12]).to(device)

        #self.out = nn.Linear(hidden_dim, out_channels)

    def mv_gnn(self, view_d, edge_list, conv):
        emb_list = []
        for i in range(self.view_num):
            x, edge_index = view_d[i], edge_list[i]
            x = conv[i](x, edge_index)
            x = F.relu(x)
            emb_list.append(x)
        emb_view = torch.stack(emb_list, dim=0)
        return emb_view
    def forward(self, view_data):
        edge_list = [view_data[i].edge_index for i in range(self.view_num)]  #这里是整图的多视角图数据！！！！！！！！！
        view_data_in = [view_data[i].x for i in range(self.view_num)]         #这里是整图的多视角图数据！！！！！！！！！！！
        # [num_views, num_nodes, num_features]

        # 第一层
        emb_view = self.mv_gnn(view_data_in, edge_list, self.conv1)
        emb_view = self.dropout(emb_view)
        #emb_view=self.layer_norm(emb_view)
        # [num_views, num_nodes, emb_dim]
        attn_output = self.encoder_layer(emb_view)


        # 第二层
        emb_view_2 = self.mv_gnn(attn_output, edge_list, self.conv2)
        emb_view_2 = self.dropout(emb_view_2)
        #emb_view_2 = self.layer_norm(emb_view_2)
        attn_output1 = self.encoder_layer1(emb_view_2)

        # attn_output1 = attn_output1 + attn_output

        # 全局特征融合
        global_emb, att = self.attention(attn_output1)

        # new_features_list = [global_emb[idx_range.start:idx_range.stop] for idx_range in self.ranges]
        # h = torch.cat(new_features_list,dim=0).to(device)
        logits,_ = self.myGATconv(self.g , global_emb,res_attn = None)
        logits = logits.mean(1)
        logits = logits / torch.max(torch.norm(logits,dim=1,keepdim=True),self.epsilon)

        # global_emb = torch.mean(attn_output1, dim=0)
        #output = self.out(global_emb)
        return  [attn_output, attn_output1], logits,global_emb

        #return output, [attn_output, attn_output1], global_emb


class MLPBlock(nn.Sequential):
    """Transformer MLP block."""

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, mlp_dim)
        self.act = nn.GELU()
        self.dropout_1 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(mlp_dim, in_dim)
        self.dropout_2 = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.normal_(self.linear_1.bias, std=1e-6)
        nn.init.normal_(self.linear_2.bias, std=1e-6)
class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(query=x, key=x, value=x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


# pylint: enable=W0235
import torch as th
from torch import nn
import torch.nn.functional as F
from dgl import function as fn
from dgl.nn.pytorch import edge_softmax
from dgl._ffi.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair
class myGATConv(nn.Module):
    """
    Adapted from
    https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/gatconv.html#GATConv
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=False,
                 alpha=0.):
        super(myGATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
        self.bias = bias
        if bias:
            self.bias_param = nn.Parameter(th.zeros((1, num_heads, out_feats)))
        self.alpha = alpha

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, res_attn=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    -1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]

            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))

            e = self.leaky_relu(graph.edata.pop('e'))

            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            if res_attn is not None:
                graph.edata['a'] = graph.edata['a'] * (1-self.alpha) + res_attn * self.alpha #alpha是beta
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            # bias
            if self.bias:
                rst = rst + self.bias_param
            # activation
            if self.activation:
                rst = self.activation(rst)
            return rst, graph.edata.pop('a').detach()

class preGATConv(nn.Module):
    # 去掉了激活函数
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True):
        super(preGATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.fc = nn.Linear(
            self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        # 原论文如此
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if bias:
            self.bias = nn.Parameter(th.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer('bias', None)
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
        self.alpha = 0.5

    def reset_parameters(self):

        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)


    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value


    def forward(self, graph, feat, get_attention=False):
        src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
        h_src = h_dst = self.feat_drop(feat)
        feat_src = feat_dst = self.fc(h_src).view(
            *src_prefix_shape, self._num_heads, self._out_feats)
        if graph.is_block:
            feat_dst = feat_src[:graph.number_of_dst_nodes()]
            h_dst = h_dst[:graph.number_of_dst_nodes()]
            dst_prefix_shape = (graph.number_of_dst_nodes(),) + dst_prefix_shape[1:]
        el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
        graph.srcdata.update({'ft': feat_src, 'el': el})
        graph.dstdata.update({'er': er})
       # 计算
        # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
        graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
        # e = self.leaky_relu(graph.edata.pop('e'))
        e = self.leaky_relu(graph.edata.pop('e'))
        # compute softmax
        graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
        w = graph.edata['w']
        graph.edata['w'] = edge_softmax(graph, w)
        # graph.edata['nw'] = graph.edata['nw'].unsqueeze(1).unsqueeze(1)
        # graph.edata['nw'] = graph.edata['nw'].repeat(graph.edata['a'].shape[0] // graph.edata['nw'].shape[0],
        #                                            graph.edata['a'].shape[1] // graph.edata['nw'].shape[1],
        #                                            graph.edata['a'].shape[2] // graph.edata['nw'].shape[2])
        # print("后")
        # print(graph.edata['nw'].shape)
        # print(graph.edata['a'].shape)
        # exit(0)
        graph.edata['a'] = graph.edata['a'].squeeze(1).squeeze(1)
        # print(graph.edata['a'].shape)
        # exit(0)
        graph.edata['a'] = graph.edata['a'] * (1 - self.alpha) + graph.edata['w']*  self.alpha
        # graph.edata['a'] = graph.edata['a'] * (1 - self.alpha)

        graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                         fn.sum('m', 'ft'))
        rst = graph.dstdata['ft']

        # residual
        if self.res_fc is not None:
            # Use -1 rather than self._num_heads to handle broadcasting
            # 注意这边家的是dst 因为对于有向图来说之后dst会被更新 但是这边的residual没加ε
            resval = self.res_fc(h_dst).view(*dst_prefix_shape, -1, self._out_feats)
            rst = rst + resval
        # bias
        if self.bias is not None:
            rst = rst + self.bias.view(
                *((1,) * len(dst_prefix_shape)), self._num_heads, self._out_feats)
        # activation
        if self.activation:
            rst = self.activation(rst)

        if get_attention:
            return rst, graph.edata['a']
        else:
            return rst


class preGATConvHereo(nn.Module):
    # 目前就是一个最朴素的gat
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True):
        super(preGATConvHereo, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.fc = nn.Linear(
            self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        # 原论文如此
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if bias:
            self.bias = nn.Parameter(th.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer('bias', None)
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
        self.tanh = nn.Tanh()
        self.eps = 0.3

    def reset_parameters(self):

        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)


    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value


    def forward(self, graph, feat, get_attention=False):
        src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
        h_src = h_dst = self.feat_drop(feat)
        feat_src = feat_dst = self.fc(h_src).view(
            *src_prefix_shape, self._num_heads, self._out_feats)
        if graph.is_block:
            feat_dst = feat_src[:graph.number_of_dst_nodes()]
            h_dst = h_dst[:graph.number_of_dst_nodes()]
            dst_prefix_shape = (graph.number_of_dst_nodes(),) + dst_prefix_shape[1:]
        #left正则化

        degs = graph.out_degrees().float().clamp(min=1)
        norm = th.pow(degs, -0.5)
        shp = norm.shape + (1,) * (feat_src.dim() - 1)
        norm = th.reshape(norm, shp)
        feat_src = feat_src * norm

        el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
        graph.srcdata.update({'ft': feat_src, 'el': el})
        graph.dstdata.update({'er': er})
       # 计算
        # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
        graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
        # 去掉softmax
        e = self.tanh(graph.edata.pop('e'))

        # compute softmax
        # e = graph.edata.pop('e')

        #去掉softmax
        graph.edata['a'] = self.attn_drop(e)

        # compute softmax
        # graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))

        graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                         fn.sum('m', 'ft'))
        rst = graph.dstdata['ft']
        # right正则化

        degs = graph.in_degrees().float().clamp(min=1)
        norm = th.pow(degs, -0.5)
        shp = norm.shape + (1,) * (feat_dst.dim() - 1)
        norm = th.reshape(norm, shp)
        rst = rst * norm

        # residual
        if self.res_fc is not None:
            # Use -1 rather than self._num_heads to handle broadcasting
            # 注意这边家的是dst 因为对于有向图来说之后dst会被更新 但是这边的residual没加ε
            resval = self.res_fc(h_dst).view(*dst_prefix_shape, -1, self._out_feats)
            rst = rst + resval*self.eps
        # bias
        if self.bias is not None:
            rst = rst + self.bias.view(
                *((1,) * len(dst_prefix_shape)), self._num_heads, self._out_feats)
        # activation
        if self.activation:
            rst = self.activation(rst)

        if get_attention:
            return rst, graph.edata['a']
        else:
            return rst

class preGATConvWo(nn.Module):
    """
    Adapted from
    https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/gatconv.html#GATConv
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=False,
                 alpha=0.):
        super(preGATConvWo, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
        self.bias = bias
        if bias:
            self.bias_param = nn.Parameter(th.zeros((1, num_heads, out_feats)))
        self.alpha = alpha

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, res_attn=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    -1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]

            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})

            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            if res_attn is not None:
                graph.edata['a'] = graph.edata['a'] * (1-self.alpha) + res_attn * self.alpha #a是注意力alpha
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            # bias
            if self.bias:
                rst = rst + self.bias_param
            # activation
            if self.activation:
                rst = self.activation(rst)
            return rst, graph.edata.pop('a').detach()


class FALayer(nn.Module):
    # input h of all node
    # return the later item
    def __init__(self, g, in_dim, dropout):
        super(FALayer, self).__init__()
        self.g = g
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(2 * in_dim, 1)
        self.leaky = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()
        nn.init.xavier_normal_(self.gate.weight, gain=1.414)

    def edge_applying(self, edges):
        h2 = th.cat([edges.dst['h'], edges.src['h']], dim=1)
        # 换成self.leaky效果竟然还好 震惊 就是说本文和GAT的差距主要在于没有softmax
        g = self.tanh(self.gate(h2)).squeeze() # g是没正则化的 估计之前用过效果不好 这边这个squeeze是吧维度唯一的删除 比如N*1 变成 N
        # 换成GAT torch.tanh
        # g = self.dropout(g)
        # 这个d应该是度数的1/2次方
        e = g * edges.dst['d'] * edges.src['d']
        e = self.dropout(e)
        return {'e': e, 'm': g}

    def forward(self, h):
        self.g.ndata['h'] = h # 将特征赋给图 h[0]必须等于点的数量
        self.g.apply_edges(self.edge_applying)
        self.g.update_all(fn.u_mul_e('h', 'e', '_'), fn.sum('_', 'z'))

        return self.g.ndata['z']


class FAGCN(nn.Module):
    # def __init__(self, g, in_dim, hidden_dim, out_dim, dropout, eps, layer_num=2):
    def __init__(self, g, hidden_dim, out_dim, dropout, eps, layer_num=2):
        super(FAGCN, self).__init__()
        self.g = g
        self.eps = eps
        self.layer_num = layer_num
        self.dropout = dropout

        self.layers = nn.ModuleList()
        for i in range(self.layer_num):
            self.layers.append(FALayer(self.g, hidden_dim, dropout))

        # self.t1 = nn.Linear(in_dim, hidden_dim)
        self.t2 = nn.Linear(hidden_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        nn.init.xavier_normal_(self.t2.weight, gain=1.414)

    def forward(self, h):
        # h = F.dropout(h, p=self.dropout, training=self.training)
        h = th.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        raw = h
        for i in range(self.layer_num):
            h = self.layers[i](h)
            h = self.eps * raw + h
        h = self.t2(h)
        return h

class preGATConvPrior(nn.Module):

    def __init__(self,
                 in_feats, #F1
                 out_feats, # F2
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True):
        super(preGATConvPrior, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        # 判断传入的in_feat是否是tuple 其实就是判断传入的是否是异构图  下同
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        # 原论文如此
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if bias:
            self.bias = nn.Parameter(th.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer('bias', None)
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)


    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, get_attention=False):

        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            # if isinstance(feat, tuple):
            #     src_prefix_shape = feat[0].shape[:-1]
            #     dst_prefix_shape = feat[1].shape[:-1]
            #     h_src = self.feat_drop(feat[0])
            #     h_dst = self.feat_drop(feat[1])
            #     if not hasattr(self, 'fc_src'):
            #         feat_src = self.fc(h_src).view(
            #             *src_prefix_shape, self._num_heads, self._out_feats)
            #         feat_dst = self.fc(h_dst).view(
            #             *dst_prefix_shape, self._num_heads, self._out_feats)
            #     else:
            #         feat_src = self.fc_src(h_src).view(
            #             *src_prefix_shape, self._num_heads, self._out_feats)
            #         feat_dst = self.fc_dst(h_dst).view(
            #             *dst_prefix_shape, self._num_heads, self._out_feats)
            # else:
            src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
            h_src = h_dst = self.feat_drop(feat)
            feat_src = feat_dst = self.fc(h_src).view(
                *src_prefix_shape, self._num_heads, self._out_feats)
            if graph.is_block:
                feat_dst = feat_src[:graph.number_of_dst_nodes()]
                h_dst = h_dst[:graph.number_of_dst_nodes()]
                dst_prefix_shape = (graph.number_of_dst_nodes(),) + dst_prefix_shape[1:]
            # el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            # er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src})
            # graph.srcdata.update({'ft': feat_src, 'el': el})
            # graph.dstdata.update({'er': er})
            # # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            # graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            # e = self.leaky_relu(graph.edata.pop('w'))
            e = graph.edata.pop('w')
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            # message passing
            # sum是update函数 对某一维求和（1其实0也一样） 之后得到N*F的矩阵 表示每个点的特征
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                # Use -1 rather than self._num_heads to handle broadcasting
                # 注意这边家的是dst 因为对于有向图来说之后dst会被更新 但是这边的residual没加ε
                resval = self.res_fc(h_dst).view(*dst_prefix_shape, -1, self._out_feats)
                rst = rst + resval
            # bias
            if self.bias is not None:
                rst = rst + self.bias.view(
                    *((1,) * len(dst_prefix_shape)), self._num_heads, self._out_feats)
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst


class preGATConvMixHop(nn.Module):
    # 没有残差
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True):
        super(preGATConvMixHop, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.fc = nn.Linear(
            self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        # 原论文如此
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if bias:
            self.bias = nn.Parameter(th.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer('bias', None)
        # if residual:
        #     if self._in_dst_feats != out_feats * num_heads:
        #         self.res_fc = nn.Linear(
        #             self._in_dst_feats, num_heads * out_feats, bias=False)
        #     else:
        #         self.res_fc = Identity()
        # else:
        #     self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
        self.tanh = nn.Tanh()

    def reset_parameters(self):

        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        # if isinstance(self.res_fc, nn.Linear):
        #     nn.init.xavier_normal_(self.res_fc.weight, gain=gain)


    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value


    def forward(self, graph, feat, get_attention=False):
        src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
        h_src = h_dst = self.feat_drop(feat)
        feat_src = feat_dst = self.fc(h_src).view(
            *src_prefix_shape, self._num_heads, self._out_feats)
        if graph.is_block:
            feat_dst = feat_src[:graph.number_of_dst_nodes()]
            h_dst = h_dst[:graph.number_of_dst_nodes()]
            dst_prefix_shape = (graph.number_of_dst_nodes(),) + dst_prefix_shape[1:]
        el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
        graph.srcdata.update({'ft': feat_src, 'el': el})
        graph.dstdata.update({'er': er})
       # 计算
        # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
        graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
        e = self.leaky_relu(graph.edata.pop('e'))
        # compute softmax
        graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))

        graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                         fn.sum('m', 'ft'))
        rst = graph.dstdata['ft']

        # residual
        # bias
        if self.bias is not None:
            rst = rst + self.bias.view(
                *((1,) * len(dst_prefix_shape)), self._num_heads, self._out_feats)
        # activation
        if self.activation:
            rst = self.activation(rst)

        if get_attention:
            return rst, graph.edata['a']
        else:
            return rst


class preGATConvMixHopCutNode(nn.Module):
    # 加上残差
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True,
                 residual = False):
        super(preGATConvMixHopCutNode, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.fc = nn.Linear(
            self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        # 原论文如此
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if bias:
            self.bias = nn.Parameter(th.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer('bias', None)
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
        self.tanh = nn.Tanh()

    def reset_parameters(self):

        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)


    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value


    def forward(self, graph, feat, get_attention=False):
        src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
        h_src = h_dst = self.feat_drop(feat)
        feat_src = feat_dst = self.fc(h_src).view(
            *src_prefix_shape, self._num_heads, self._out_feats)
        if graph.is_block:
            feat_dst = feat_src[:graph.number_of_dst_nodes()]
            h_dst = h_dst[:graph.number_of_dst_nodes()]
            dst_prefix_shape = (graph.number_of_dst_nodes(),) + dst_prefix_shape[1:]
        el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
        graph.srcdata.update({'ft': feat_src, 'el': el})
        graph.dstdata.update({'er': er})
       # 计算
        # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
        graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
        e = self.leaky_relu(graph.edata.pop('e'))
        # e = graph.edata.pop('e')
        # compute softmax
        graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))

        e = graph.edata.pop('a')
        zero = th.zeros(e.shape).to(th.device('cuda:2'))
        e = th.where(e < 0.25, zero, e)
        graph.edata['a'] = e

        graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                         fn.sum('m', 'ft'))
        rst = graph.dstdata['ft']

        # residual
        if self.res_fc is not None:
            resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
            rst = rst + resval
        # bias
        if self.bias is not None:
            rst = rst + self.bias.view(
                *((1,) * len(dst_prefix_shape)), self._num_heads, self._out_feats)
        # activation
        if self.activation:
            rst = self.activation(rst)

        if get_attention:
            return rst, graph.edata['a']
        else:
            return rst