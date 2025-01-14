import torch.nn as nn
import math
import torch
import torch.nn.functional as F
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from itertools import accumulate

import torch
import dgl
from torch_geometric.nn import knn_graph
from utils.data_loader import *
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from DP_AC_transformer_aggregate_frame_noHGNN import *

from DP_AC_Attribute_propagation_within_the_diffusion_path import *
from models.model_manager import *
from utils.data_process import *
from utils.tools import *
from scipy.sparse import dia_matrix
from collections import deque
from scipy.sparse import csr_matrix
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
from retrainer1 import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import numpy as np
from scipy.optimize import minimize
from torch.backends import cudnn
from FixedNet2 import *
import tracemalloc
from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import torch as th
from torch import nn
import torch.nn.functional as F
from dgl import function as fn
from dgl.nn.pytorch import edge_softmax
from dgl._ffi.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_sparse import coalesce
#vision
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

#消融实验
from PPNP import *
from GCN import *
from Mean import *
class MDNNModel_5(nn.Module):
    def __init__(self,hgs,split_list,heads,select,select_1,hgs_1,
                 ranges,args,data_info, idx_info,train_info,
                 adjM,labels,dl,type_mask,train_val_test_idx,select_2,hgs_2,select_3,hgs_3 ,cur_repeat,original_feature):
        super(MDNNModel_5, self).__init__()
        self.cur_repeat =  cur_repeat
        self.original_feature=original_feature
        # self.x=x
        self.args = args

        self.train_idx, self.val_idx, self.test_idx = idx_info
        self._criterion = train_info
        self.labels = labels
        self.type_mask=type_mask
        self.train_val_test_idx=train_val_test_idx
        self.dl=dl
        self.adjM = adjM

        self._logger = args.logger
        self.data_info = data_info
        self.idx_info = idx_info
        self.features_list, self.labels, self.g, self.type_mask, self.dl, self.in_dims, self.num_classes = data_info
        self.train_idx, self.val_idx, self.test_idx = idx_info
        self._criterion = train_info
        self.ranges=ranges
        self.dl = dl
        self._data_info = None
        self._infer_new_features_list =None
        self._writer=None


        self.split_list = split_list
        self.heads = heads

        self.hgs = hgs
        self.hgs_1 = hgs_1
        self.hgs_2 = hgs_2
        self.hgs_3 = hgs_3
        self.select = select
        self.select_1=select_1
        self.select_2 = select_2
        self.select_3 = select_3

        self.net = myGAT(data_info[2], hgs, data_info[5], self.args.attn_vec_dim, data_info[6],
                         self.args.complete_num_layers, self.args.intralayers, heads,
                         F.elu, self.args.dropout, self.args.dropout, self.args.slope, True, 0.00)
        self.net_1 = myGAT(data_info[2], hgs_1, data_info[5], self.args.attn_vec_dim, data_info[6],
                         self.args.complete_num_layers, self.args.intralayers, heads,
                         F.elu, self.args.dropout, self.args.dropout, self.args.slope, True, 0.00)
        self.net_2 = myGAT(data_info[2], hgs_2, data_info[5], self.args.attn_vec_dim, data_info[6],
                         self.args.complete_num_layers, self.args.intralayers, heads,
                         F.elu, self.args.dropout, self.args.dropout, self.args.slope, True, 0.00)
        # self.net_3 = myGAT(data_info[2], hgs_3, data_info[5], self.args.attn_vec_dim, data_info[6],
        #                    self.args.complete_num_layers, self.args.intralayers, heads,
        #                    F.elu, self.args.dropout, self.args.dropout, self.args.slope, True, 0.00)


        self.net.to(device)
        self.net_1.to(device)
        self.net_2.to(device)
        # self.net_3.to(device)



        self.num_features_list = [self.args.attn_vec_dim   for _ in
                                  range(self.args.max_num_views)]
        self.multi_view_interaction_model = M_GCN_t_noHGNN(self.num_features_list, hidden_dim=args.attn_vec_dim,args = args,dl=dl,data_info=data_info).to(
            device)  # self.args.max_features_len
        self.hgnn_model_manager = ModelManager(data_info, idx_info, args)
        # 创建 GNN 模型
        self.hgnn_model = self.hgnn_model_manager.create_model_class().to(device)
        # save_dir = save_dir_name(self.args)  # 调用函数获取路径
        # self._writer = SummaryWriter(f'/home/yyj/MDNN-AC/AutoAC-main/tf-logs/{save_dir}')

        self.hgnn_preprocess = nn.Linear(args.attn_vec_dim, args.hidden_dim, bias=True).to(
            device)  # args.max_features_len
        nn.init.xavier_normal_(self.hgnn_preprocess.weight, gain=1.414).to(device)
        self._save_dir = save_dir_name(self.args)  # 调用函数获取路径
        self._writer = SummaryWriter(f'/home/yyj/MDNN-AC/AutoAC-main/tf-logs/{self._save_dir}')
        self.fc_list = nn.ModuleList(
            [nn.Linear(in_dim, self.args.attn_vec_dim, bias=True) for in_dim in self.data_info[5]]).to(device)
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        #测试消融实验
        # 基于PPNP补全：
        # 初始化 PPNP 模型
        self.model1 = PPNP1(in_channels=self.args.attn_vec_dim, out_channels=self.args.attn_vec_dim, alpha=0.1).to(
            device)
        self.model2 = PPNP2(in_channels=self.args.attn_vec_dim, out_channels=self.args.attn_vec_dim, alpha=0.1).to(
            device)

        # 基于GCN补全：
        # 初始化 GCN 模型
        self.GCN_model1 = GCN1(in_channels=self.args.attn_vec_dim, hidden_channels=self.args.hidden_dim,
                               out_channels=self.args.attn_vec_dim).to(
            device)
        self.GCN_model2 = GCN2(in_channels=self.args.attn_vec_dim, hidden_channels=self.args.hidden_dim,
                               out_channels=self.args.attn_vec_dim).to(
            device)

    def create_retrain_model(self, new_features_list,new_data_info, new_idx_info,new_train_info):  # 用于创建一个用于重新训练的模型。它接受两个参数 alpha 和 node_assign，然后基于这些参数创建一个新的 FixedNet 模型。
        inner_data_info = self.hgnn_model_manager.get_graph_info()
        gnn_model_manager = self.hgnn_model_manager
        model = FixedNet2(new_features_list,new_data_info, new_idx_info, new_train_info, inner_data_info, gnn_model_manager, self.args)

        return model

    def forward(self):


        #补全之后的第一個視角
        feat = self.net(self.select,self.split_list,self.features_list)
        # #补全之后的第二個視角
        feat_aug_1 = self.net_1(self.select_1,self.split_list,self.features_list)

        #第三个视角
        feat_aug_2 = self.net_2(self.select_2,self.split_list,self.features_list)
        # # 第四个视角
        # feat_aug_3 = self.net_3(self.select_3, self.split_list, self.features_list)
        # #4
        # feat_aug_fourth = self.model(self.x_prop_aug_fourth.to(device))
        # #5
        # feat_aug_fifth = self.model(self.x_prop_aug_fifth.to(device))
        # #6
        # feat_aug_sixth = self.model(self.x_prop_aug_sixth.to(device))
        '''用于测试消融实验'''
        # h = []
        # for fc, feature in zip(self.fc_list, self.data_info[0]):
        #     h.append(fc(feature.to(device)))
        # h1 = h.copy()
        # h2 = h.copy()
        # ####第一个：基于PPNP的方法进行补全：
        # #第一个视图
        # for i in range(len(self.hgs)):
        #     preh1 = h1
        #     #all
        #     ph1 = []
        #     s= int(self.select[i])
        #     h1 = torch.cat(h1[:s+1],0)
        #
        #     out1 = self.model1(h1, torch.stack(self.hgs[i].edges(),0).to(device))
        #     # 使用PPNP输出对缺失值进行补全
        #     x_filled_1 = ppnp_based_aggregation(h1.clone(), out1)
        #
        #     ph1 = torch.split(x_filled_1,self.split_list[:s+1],0)
        #     preh1[s] = ph1[s]
        #     h1 = preh1
        # new_h= torch.cat(h1,0)
        # new_h= new_h.squeeze()
        #
        # # 第二个视图
        # for i in range(len(self.hgs_1)):
        #     preh2 = h2
        #     # all
        #     ph2 = []
        #     s = int(self.select_1[i])
        #     h2 = torch.cat(h2[:s + 1], 0)
        #
        #     out2 = self.model2(h2, torch.stack(self.hgs_1[i].edges(), 0).to(device))
        #     # 使用PPNP输出对缺失值进行补全
        #     x_filled_2 = ppnp_based_aggregation(h2.clone(), out2)
        #
        #     ph2 = torch.split(x_filled_2, self.split_list[ :s + 1], 0)
        #     preh2[s] = ph2[s]
        #     h2 = preh2
        # new_h_2 = torch.cat(h2, 0)
        # new_h_2 = new_h_2.squeeze()

        # ###第二个：基于GCN的方法进行补全：
        # #第一个视图
        # for i in range(len(self.hgs)):
        #     preh1 = h1
        #     # all
        #     ph1 = []
        #     s = int(self.select[i])
        #     h1 = torch.cat(h1[:s + 1], 0)
        #
        #     out1 = self.GCN_model1(h1, torch.stack(self.hgs[i].edges(), 0).to(device))
        #     # 使用GCN输出对缺失值进行补全
        #     x_filled_1 = gcn_based_aggregation(h1.clone(), out1)
        #
        #     ph1 = torch.split(x_filled_1, self.split_list[:s + 1], 0)
        #     preh1[s] = ph1[s]
        #     h1 = preh1
        # new_h = torch.cat(h1, 0)
        # new_h = new_h.squeeze()
        #
        # # 第二个视图
        # for i in range(len(self.hgs_1)):
        #     preh2 = h2
        #     # all
        #     ph2 = []
        #     s = int(self.select_1[i])
        #     h2 = torch.cat(h2[:s + 1], 0)
        #
        #     out2 = self.GCN_model2(h2, torch.stack(self.hgs_1[i].edges(), 0).to(device))
        #     # 使用GCN输出对缺失值进行补全
        #     x_filled_2 = gcn_based_aggregation(h2.clone(), out2)
        #
        #     ph2 = torch.split(x_filled_2, self.split_list[:s + 1], 0)
        #     preh2[s] = ph2[s]
        #     h2 = preh2
        # new_h_2 = torch.cat(h2, 0)
        # new_h_2 = new_h_2.squeeze()
        #
        # ####第三个：基于Mean的方法进行补全：
        # # 第一个视图
        # for i in range(len(self.hgs)):
        #     preh1 = h1
        #     # all
        #     ph1 = []
        #     s = int(self.select[i])
        #     h1 = torch.cat(h1[:s + 1], 0)
        #
        #     x_filled_1 = mean_attribute_aggregation(h1.clone(), self.hgs[i].edges())
        #
        #     ph1 = torch.split(x_filled_1, self.split_list[:s + 1], 0)
        #     preh1[s] = ph1[s]
        #     h1 = preh1
        # new_h = torch.cat(h1, 0)
        # new_h = new_h.squeeze()
        #
        # # 第二个视图
        # for i in range(len(self.hgs_1)):
        #     preh2 = h2
        #     # all
        #     ph2 = []
        #     s = int(self.select_1[i])
        #     h2 = torch.cat(h2[:s + 1], 0)
        #
        #     x_filled_2 = mean_attribute_aggregation(h2.clone(), self.hgs_1[i].edges())
        #
        #     ph2 = torch.split(x_filled_2, self.split_list[:s + 1], 0)
        #     preh2[s] = ph2[s]
        #     h2 = preh2
        # new_h_2 = torch.cat(h2, 0)
        # new_h_2 = new_h_2.squeeze()
        # views_tensors = []
        #
        # views_tensors.append(new_h.to(device))
        # views_tensors.append(new_h_2.to(device))
        '''结束测试消融实验'''
        views_tensors = []
        views_tensors.append(feat)
        views_tensors.append(feat_aug_1)
        views_tensors.append(feat_aug_2)
        # views_tensors.append(feat_aug_3)



        '''# # 复制原始的邻接矩阵 self.args.max_num_views 次'''
        edge_index_list = [torch.stack(self.data_info[2].edges(), dim=0) for _ in
                           range(self.args.max_num_views)]#不知为啥这个适用于dblp数据集


        # 构造 view_data
        view_data_list = []
        for i in range(self.args.max_num_views):
            view_data = Data(x=views_tensors[i], edge_index=edge_index_list[i])
            view_data_list.append(view_data)
        emb_view_layer, logits, completed_features= self.multi_view_interaction_model(view_data_list)

        '''========基于注意力机制的全局特征融合/基于扩散路径的节点属性聚合框架============================================================================================================================================================================='''
        '''这里推断其实已经结束，那就只剩构建特征以便于计算损失！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！'''
        return  emb_view_layer, logits,completed_features

        # completed_features为要输入下游任务的更新后的features_list、、
        # logits  这用于计算下游损失
        # emb_view_layer   这用于构建多视角特征
    def tranin_and_val(self, model,mini_batch_input=None):
        optimizer1 = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        bst_val_loss = [np.inf]  # 使用列表包装，使其成为可变对象
        _earlystop = EarlyStopping_Search(logger=self.args.logger,
                                          patience=self.args.patience_retrain)  # args.patience)#_search)

        for epoch in range(self.args.search_epoch):
            t_start = time.time()
            model.train()
            optimizer1.zero_grad()

            # 前向传播
            multi_view_nodes,logits,_ = model.forward()

            _y = self.labels
            if self.args.dataset == 'IMDB':
                self.target = torch.FloatTensor(_y).to(device)
            else:
                self.target = torch.LongTensor(_y).to(device)

            if self.args.dataset == 'IMDB':
                logits = torch.sigmoid(logits).to(device)  # softmax sigmoid
            # Calculate train loss
            logits_train = logits[self.train_idx].to(device)
            target_train = self.target[self.train_idx]


            # # Calculate train loss
            train_loss = self._criterion(logits_train, target_train)

            #Calculate consistency loss 多视角启动
            loss_consistency1 = loss_each_view(multi_view_nodes[1])
            # loss_consistency2 = Diversity_loss(multi_view_nodes[0])


            loss = train_loss + self.args.beta_1 * loss_consistency1 #+ (1-self.args.beta_1) * loss_consistency2
            # loss = train_loss

            loss.backward()  # retain_graph=True)
            nn.utils.clip_grad_norm_(model.parameters(), self.args.grad_clip)  #然后还要补充args.grad_clip！！
            optimizer1.step()
            # model.lr_step(epoch)
            # scheduler.step()
            t_train = time.time()


            model.eval()
            with torch.no_grad():
                # 前向传播
                _, infer_logits,_ = model.forward()

                _infer_y = self.labels
                if self.args.dataset == 'IMDB':
                    self.infer_target = torch.FloatTensor(_infer_y).to(device)
                else:
                    self.infer_target = torch.LongTensor(_infer_y).to(device)

                if self.args.dataset == 'IMDB':
                    infer_logits = torch.sigmoid(infer_logits).to(device)
                logits_val = infer_logits[self.val_idx].to(device)

                val_loss = self._criterion(logits_val, self.infer_target[self.val_idx])


            t_end = time.time()
            if is_save(bst_val_loss, loss, val_loss):
                 save_search_info(model, self.args)

            #多视角
            self._logger.info(
                'Epoch_batch_{:05d} | lr {:.4f} |Train_Loss {:.4f} |  total_loss_consistency1 {:.4f} | Loss {:.4f} | Val_Loss {:.4f}| Train Time(s) {:.4f}| Val Time(s) {:.4f} | Time(s) {:.4f}'.format(
                    epoch, optimizer1.state_dict()['param_groups'][0]['lr'], train_loss.item(),
                    loss_consistency1.item(), loss,val_loss.item(), t_train - t_start, t_end - t_train,t_end - t_start ))  # loss_consistency2
            # #单视角
            # self._logger.info(
            #     'Epoch_batch_{:05d} | lr {:.4f} |Train_Loss {:.4f} |  Loss {:.4f} | Val_Loss {:.4f}| Train Time(s) {:.4f}| Val Time(s) {:.4f} | Time(s) {:.4f}'.format(
            #         epoch, optimizer1.state_dict()['param_groups'][0]['lr'], train_loss.item(),
            #         loss, val_loss.item(), t_train - t_start, t_end - t_train,
            #                                                          t_end - t_start))  # loss_consistency2


            self._writer.add_scalar(f'{self.args.dataset}_train_loss', loss, global_step=epoch)
            self._writer.add_scalar(f'{self.args.dataset}_val_loss', val_loss, global_step=epoch)
            # # # 这里就是保存模型的代码段！！！！！！！！！！！！！！！！！！这里要改一下

            _earlystop(loss, val_loss.item())
            if _earlystop.early_stop:
                self.args.logger.info('Eearly stopping!')
                break

        torch.cuda.empty_cache()
        gc.collect()
    def test(self, model,mini_batch_input=None):
        self._logger.info('\ntesting...')
        # 加载保存的模型参数
        model.load_state_dict(torch.load('/home/yyj/MDNN-AC/AutoAC-main/checkpoint/save/net_params_{}.pt'.format(self.args.time_line)))
        model.eval()
        # 创建提交目录
        # if not os.path.exists(f'submit/submit_{self.genotype_dir_name}_{self.args.time_line}'):
        #     os.makedirs(f'submit/submit_{self.genotype_dir_name}_{self.args.time_line}')

        if not os.path.exists(f'submit/submit_{save_dir_name(self.args)}'):
            os.makedirs(f'submit/submit_{save_dir_name(self.args)}')

        self._logger.info(f'submit dir: submit/submit_{save_dir_name(self.args)}')

        # self._logger.info(f'submit dir: submit/submit_{self.genotype_dir_name}')

        with torch.no_grad():
            if self.args.use_minibatch is False:
                multi_view_nodes, logits,completed_features = model.forward()


                logits_test = logits[self.test_idx]
            else:
                logits_test = []
                test_idx_generator = index_generator(batch_size=self.args.batch_size_test, indices=self.test_idx,
                                                     shuffle=False)
                for iteration in range(test_idx_generator.num_iterations()):
                    test_idx_batch = test_idx_generator.next()
                    test_g_list, test_indices_list, test_idx_batch_mapped_list = parse_minibatch(
                        self.adjlists, self.edge_metapath_indices_list, test_idx_batch, device,
                        self.args.neighbor_samples)
                    multi_view_nodes, logits, completed_features= model(self.combined_features, (
                        test_g_list, test_indices_list, test_idx_batch_mapped_list, test_idx_batch))
                    logits_test.append(logits)

                logits_test = torch.cat(logits_test, 0).to(device)

            if self.args.dataset == 'IMDB':
                pred = (logits_test.cpu().numpy() > 0).astype(int)
                # self.dl.gen_file_for_evaluate(test_idx=self.test_idx, label=pred,
                #                             file_path=(f'submit/submit_{self.genotype_dir_name}_{self.args.time_line}/{self.args.dataset}_{cur_repeat + 1}.txt'), mode='multi')
                # self.dl.gen_file_for_evaluate(test_idx=self.test_idx, label=pred,file_path=('submit/submit_{self.genotype_dir_name}/{self.args.dataset}_{cur_repeat + 1}.txt'),mode='multi')
                self.dl.gen_file_for_evaluate(test_idx=self.test_idx, label=pred, file_path=(
                    f'submit/submit_{save_dir_name(self.args)}/{self.args.dataset}_{self.cur_repeat+1}.txt'), mode='multi')

                self._logger.info(self.dl.evaluate(pred))

            else:
                pred = logits_test.cpu().numpy().argmax(axis=1)
                # self.dl.gen_file_for_evaluate(test_idx=self.test_idx, label=pred,
                #                             file_path=(f'submit/submit_{self.genotype_dir_name}_{self.args.time_line}/{self.args.dataset}_{cur_repeat + 1}.txt'))
                # self.dl.gen_file_for_evaluate(test_idx=self.test_idx, label=pred,file_path=(f'submit/submit_{self.genotype_dir_name}/{self.args.dataset}_{cur_repeat + 1}.txt'))
                self.dl.gen_file_for_evaluate(test_idx=self.test_idx, label=pred, file_path=(
                    f'submit/submit_{save_dir_name(self.args)}/{self.args.dataset}_{self.cur_repeat+1}.txt'))

                onehot = np.eye(self.num_classes, dtype=np.int32)
                pred = onehot[pred]
                self._logger.info(self.dl.evaluate(pred))
        '''t-sne画图可视化
        # #t-sne画图
        original_features_np = self.original_feature[0][self.test_idx].cpu().numpy()  # 初始测试集特征
        # 加载测试集补全后的特征
        completed_features_np = completed_features[self.test_idx].cpu().numpy()  # 补全后的特征

        # 加载测试集标签
        # labels_test = self.dl.load_labels('label.dat.test')
        # y_true = labels_test['data'][labels_test['mask']]
        # train_labels_np = y_true.cpu().numpy()

        def load_original_labels(file_path):
            """
            从原始标签文件中读取节点标签信息，并返回一个字典
            """
            node_labels = {}

            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # 解析每一行数据
                    parts = line.strip().split('\t')
                    node_id = int(parts[0])  # 节点ID
                    # node_name = parts[1]  # 节点名称（不用于标签提取）
                    # node_type = int(parts[2])  # 节点类型
                    node_label = int(parts[3])  # 节点标签

                    # 将节点标签存入字典
                    node_labels[node_id] = node_label

            return node_labels

        
        def sort_labels_by_node_id(file_path):
            """
            加载原始标签文件并根据节点 ID 从小到大排序
            """
            original_labels = load_original_labels(file_path)

            # 按照节点 ID（字典的键）从小到大排序
            sorted_labels = {k: original_labels[k] for k in sorted(original_labels.keys())}

            # 转换为与 test_feature 对应的标签数组
            labels_array = list(sorted_labels.values())
            return labels_array

        original_labels = sort_labels_by_node_id(os.path.join(self.dl.path, 'label.dat.test'))
        # # 加载测试集补全后的标签
        # completed_labels = pred.cpu().numpy()

        # def load_completions(file_path, num_nodes):
        #     """
        #     从补全后的文件中读取节点标签信息，并返回一个字典
        #     """
        #     node_labels = {}
        #
        #     with open(file_path, 'r', encoding='utf-8') as f:
        #         for line in f:
        #             # 解析每一行数据
        #             parts = line.strip().split('\t')
        #             node_id = int(parts[0])  # 节点ID
        #             # node_type = int(parts[2])  # 节点类型
        #             node_label = int(parts[3])  # 节点标签
        #
        #             # 将节点标签存入字典
        #             node_labels[node_id] = node_label
        #     # 创建一个与 test_feature 相同长度的标签数组
        #     labels_array = [node_labels.get(i) for i in range(num_nodes)]  # -1表示未标记的节点
        #
        #     return labels_array
        #
        # completed_node_labels = load_completions(
        #      f'submit/submit_{save_dir_name(self.args)}/{self.args.dataset}_{self.cur_repeat+1}.txt', num_nodes)
        # 使用t-SNE降维
        def plot_tsne(features, labels, title,filename):
            tsne = TSNE(n_components=2, random_state=42)
            reduced_features = tsne.fit_transform(features)

            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=reduced_features[:, 0], y=reduced_features[:, 1], hue=labels, palette="viridis", s=60,
                            legend='full')
            plt.title(title)
            plt.xlabel("t-SNE 1")
            plt.ylabel("t-SNE 2")
            plt.legend(title='Class')
            plt.savefig(filename,format='pdf',bbox_inches="tight")
            plt.close()  # 避免占用内存
            # plt.show()

        plot_tsne(original_features_np, original_labels, "t-SNE: Original Node Features (Before Completion)",
                  f"Original_Node_Features_information_Vision_{self.cur_repeat+1}_{self.args.dataset}.pdf")

        # 2. 使用补全后的特征和标签的可视化
        plot_tsne(completed_features_np, original_labels, "t-SNE: Completed Node Features (After Completion)",
                  f"Completed_Node_Features_information_Vision_{self.cur_repeat+1}_{self.args.dataset}.pdf")
'''
    def lr_step(self,epoch):
        self.lr_scheduler.step(epoch)

    # 在属性定义中，我们使用了@property装饰器来创建一个getter方法，
    # 然后使用.setter方法定义一个setter方法，这样我们就可以通过属性访问来设置值。
    def set_new_features_list(self,infer_new_features_list):
        self._infer_new_features_list = infer_new_features_list
    def get_new_features_list(self):
        return self._infer_new_features_list
    def set_data_info(self, new_data_info):
        self._data_info = new_data_info

    # def set_idx_info(self, new_idx_info):
    #     self._idx_info = new_idx_info
    #
    # def set_train_info(self, new_train_info):
    #     self._train_info = new_train_info

    def set_writer(self, _writer):
        self._writer = _writer

    def get_data_info(self):
        return self.data_info

    def get_idx_info(self):
        return self.idx_info

    def get_train_info(self):
        return self._criterion

    def get_writer(self):
        return self._writer

    def set_hgnn_model_manager(self, hgnn_model_manager):
        self._hgnn_model_manager = hgnn_model_manager

    def set_hgnn_model(self,hgnn_model):
        self._hgnn_model = hgnn_model


class Linear(nn.Module):
    def __init__(self, in_features, out_features, dropout, bias=False):
        super(Linear, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', a=math.sqrt(5))
        if self.bias is not None:
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        input = F.dropout(input, self.dropout, training=self.training)
        output = torch.matmul(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class MLP_encoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(MLP_encoder, self).__init__()
        self.Linear1 = Linear(nfeat, nhid, dropout, bias=True)

    def forward(self, x):
        x = torch.relu(self.Linear1(x))
        return x

class MLP_classifier(nn.Module):#
    def __init__(self, nfeat, nclass, dropout):
        super(MLP_classifier, self).__init__()
        self.Linear1 = Linear(nfeat, nclass, dropout, bias=True)

    def forward(self, x):
        out = self.Linear1(x)
        return torch.log_softmax(out, dim=1), out

class DDPT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, use_bn = False):
        super(DDPT, self).__init__()

        self.encoder = MLP_encoder(nfeat=nfeat,
                                 nhid=nhid,
                                 dropout=dropout)

        # self.classifier = MLP_classifier(nfeat=nhid,
        #                                  nclass=nclass,
        #                                  dropout=dropout)

        self.proj_head1 = Linear(nhid, nhid, dropout, bias=True)

        self.use_bn = use_bn
        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(nfeat)
            self.bn2 = nn.BatchNorm1d(nhid)

    def forward(self, features, eval = False):
        if self.use_bn:
            features = self.bn1(features)
        query_features = self.encoder(features)
        if self.use_bn:
            query_features = self.bn2(query_features)
        return query_features
        # output, emb = self.classifier(query_features)
        # if not eval:
        #     emb = self.proj_head1(query_features)
        # return emb, output



class myGAT(nn.Module):
    def __init__(self,
                 g,
                 hgs,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 intalayer,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 alpha,
                 ):
        super(myGAT, self).__init__()
        self.g = g
        self.hgs = hgs
        # self.hg2 = hg2
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.intraconvs = nn.ModuleList()
        # self.convs = nn.ModuleList()
        # self.convs2 = nn.ModuleList()
        self.activation = activation

        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        # h2gcn异配性
        # self.h2gcnfc = nn.Linear(num_hidden*3, num_hidden)
        # nn.init.xavier_normal_(self.h2gcnfc.weight, gain=1.414)
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        # 每个元路径图一个gat
        # for i in range(len(self.hgs)):
        #     self.convs.append(preGATConv(num_hidden, num_hidden, heads[-1], 0.5, 0.5, negative_slope, residual=True))
        # for i in range(len(self.hgs)):
        #     self.convs2.append(preGATConv(num_hidden, num_hidden, heads[-1], 0.5, 0.5, negative_slope, residual=True))
        for i in range(intalayer):
            temp = nn.ModuleList()
            for i in range(len(self.hgs)):
                temp.append(
                    preGATConv(num_hidden, num_hidden, heads[-1], feat_drop, attn_drop, negative_slope, residual=True))
            self.intraconvs.append(temp)
        # input projection (no residual)
        # 双层异配性
        # self.preconv = preGATConvHereo(num_hidden,num_hidden,heads[0],feat_drop,attn_drop,negative_slope,residual=True)
        # self.preconv_out = preGATConvHereo(num_hidden* heads[0],num_hidden,heads[-1],feat_drop,attn_drop,negative_slope,residual=True)
        # 单层异配性
        # self.preconv_one = preGATConvHereo(num_hidden,num_hidden,heads[-1],feat_drop,attn_drop,negative_slope,residual=True)
        # 一层GAT  原ACM
        # self.preconvacm = preGATConv(num_hidden,num_hidden,heads[-1],feat_drop,attn_drop,negative_slope,residual=True)

        # 重新实现的普通GAT
        # self.preconvacm = preGATConv(num_hidden, num_hidden, heads[0], feat_drop, attn_drop, negative_slope,
        #                           residual=True)
        # self.preconvacm1 = preGATConv(num_hidden * heads[0], num_hidden, heads[-1], feat_drop, attn_drop, negative_slope,
        #                            residual=True)

        # 普通GAT并且舍弃一些点
        # self.preconvacm = preGATConvMixHopCutNode(num_hidden, num_hidden, heads[0], feat_drop, attn_drop, negative_slope,
        #                           allow_zero_in_degree=True,residual=True)
        # self.preconvacm1 = preGATConvMixHopCutNode(num_hidden * heads[0], num_hidden, heads[-1], feat_drop, attn_drop, negative_slope,
        #                            allow_zero_in_degree=True,residual=True)

        # 利用先验卷积一层
        # self.preconvprior = preGATConvPrior(num_hidden,num_hidden,heads[-1],feat_drop,attn_drop,negative_slope,allow_zero_in_degree=True)
        # 两层先验卷积
        # self.preconvprior1 = preGATConvPrior(num_hidden, num_hidden, heads[0], feat_drop, attn_drop, negative_slope,
        #                           allow_zero_in_degree=True,residual=True)
        # self.preconvprior2 = preGATConvPrior(num_hidden * heads[0], num_hidden, heads[-1], feat_drop, attn_drop, negative_slope,
        #                            allow_zero_in_degree=True,residual=True)
        # 直接吧FAGCN生拉硬套过来
        # self.preGNN = FAGCN(self.hg,num_hidden,num_hidden,feat_drop,eps=0.3)
        # H2GCN的异配性
        # self.h2gcnconv1 = preGATConvMixHop(num_hidden,num_hidden,heads[-1],feat_drop,attn_drop,negative_slope)
        # self.h2gcnconv2 = preGATConvMixHop(num_hidden,num_hidden,heads[-1],feat_drop,attn_drop,negative_slope)
        # H2GCN删掉部分低节点
        # self.h2gcnconv1 = preGATConvMixHopCutNode(num_hidden, num_hidden, heads[-1], feat_drop, attn_drop, negative_slope)
        # self.h2gcnconv2 = preGATConvMixHopCutNode(num_hidden, num_hidden, heads[-1], feat_drop, attn_drop, negative_slope)

        self.gat_layers.append(myGATConv(
            num_hidden, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation, alpha=alpha))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(myGATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation, alpha=alpha))
        # output projection
        self.gat_layers.append(myGATConv( #* heads[-2]
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None, alpha=alpha))
        self.epsilon = torch.FloatTensor([1e-12]).cuda()

    def forward(self,select, split_list,features_list): #
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))

        # intra-type
        for i in range(len(self.hgs)):
            preh = h
            # all
            ph = []
            s = int(select[i])
            h = torch.cat(h[:s + 1], 0)  # +1是因为右边是开区间
            # print(self.hgs[i])
            # print(h.shape)
            # exit(0)
            for j in range(len(self.intraconvs)):
                h = self.intraconvs[j][i](self.hgs[i], h).flatten(1)
            # h = self.convs[i](self.hgs[i], h).flatten(1)
            # # for 2 layers
            # h = self.convs2[i](self.hgs[i], h).flatten(1)
            ph = torch.split(h, split_list[:s + 1], 0)
            preh[s] = ph[s]
            h = preh
        h = torch.cat(h, 0)
        h = h.squeeze()

        # h = torch.cat(h,0)
        # # pre层实验
        # preh = h
        # # all
        # ph = []
        # for s in select:
        #     ph.append(h[int(s)])
        # h = torch.cat(h[:int(select[-1])+1],0) # +1是因为右边是开区间
        # # H2GCN式的聚合
        # # h1 = self.h2gcnconv1(self.hg,h).squeeze()
        # # h2 = self.h2gcnconv2(self.hg2,h).squeeze()
        # # 异配性双层ACM
        # # h = self.preconv(self.hg, h).flatten(1)
        # # h = self.preconv_out(self.hg, h).squeeze()
        # # 异配性单层ACM
        # # h = self.preconv_one(self.hg, h).squeeze()
        #
        # #单层普通GAT
        # # h = self.preconvacm(self.hg,h).squeeze()
        #
        #
        # #重新实现的普通GAT
        # # h = self.preconvacm(self.hg, h).flatten(1)
        # # h = self.preconvacm1(self.hg, h).squeeze()
        #
        # # 先验一层
        # h = self.preconvacm(self.hg,h).squeeze()
        #
        # # 先验两层
        # # h = self.preconvprior1(self.hg, h).flatten(1)
        # # h = self.preconvprior2(self.hg, h).squeeze()
        #
        # # 硬套FAGCN
        # # h = self.preGNN(h)
        #
        # # 普通的聚合
        # ph = torch.split(h,split_list[:int(select[-1])+1],0)
        # for i in select:
        #     preh[int(i)] = ph[int(i)]
        # h = torch.cat(preh, 0)
        # # H2GCN式的聚合
        # # h_origin = []
        # # h_1 = []
        # # h_2 = []
        # # ph_1 = torch.split(h1, split_list[:int(select[-1]) + 1], 0)
        # # ph_2 = torch.split(h2, split_list[:int(select[-1]) + 1], 0)
        # # for i in select:
        # #     h_origin.append(preh[int(i)])
        # #     h_1.append(ph_1[int(i)])
        # #     h_2.append(ph_2[int(i)])
        # # h_origin = torch.cat(h_origin, 0)
        # # h_1 = torch.cat(h_1, 0)
        # # h_2 = torch.cat(h_2, 0)
        # # h_h2gcn = torch.cat((h_origin,h_1,h_2),1)
        # # h_h2gcn = self.h2gcnfc(h_h2gcn)
        # # # print(h_h2gcn.shape)
        # # h2gcnsplitlist = []
        # # for i in select:
        # #     h2gcnsplitlist.append(split_list[int(i)])
        # # # print(h2gcnsplitlist) #5959 1902
        # # h_h2gcn = torch.split(h_h2gcn, h2gcnsplitlist, 0)
        # # # print(h_h2gcn[0],h_h2gcn[1]) #5959 1902
        # # for index,i in enumerate(select):
        # #     # print(index,i) #0 1 // 1 3
        # #     preh[int(i)] = h_h2gcn[index]
        # # h = torch.cat(preh, 0)
        #
        # # print(h.shape)
        # # 问题是 传入pre的 h必须得有最大的那个
        # # DBLP
        # # h = torch.cat(h,0)
        # # h = self.preconv(self.hg,h).flatten(1)
        # # h = self.preconv_out(self.hg,h).squeeze()
        #
        # # ACM
        # # ph = torch.cat(h[0:2],0)
        # # ph = self.preconv(self.hg,ph).flatten(1)
        # # ph = self.preconv_out(self.hg,ph).squeeze()
        # # h = torch.cat(h, 0)
        # # h = torch.cat((ph,h[ph.shape[0]:]),0)
        # # print(h.shape)
        # # exit(0)
        # '''感覺這裡就像是論文中提到的Meta-path-free Inter-type Aggregation 無元路徑類型間聚合'''
        # res_attn = None
        # for l in range(self.num_layers):
        #     h, res_attn = self.gat_layers[l](self.g, h, res_attn=res_attn)
        #     h = h.flatten(1)#h：[26128，8，64]   变为 [26128，512]
        # # output projection
        # '''感覺像是論文中的Combination of Semantic Information'''
        # logits, _ = self.gat_layers[-1](self.g, h, res_attn=None)
        # logits = logits.mean(1)
        # # This is an equivalent replacement for tf.l2_normalize, see https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/l2_normalize for more information.
        # logits = logits / (torch.max(torch.norm(logits, dim=1, keepdim=True), self.epsilon))
        return h


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


'''自己改进的'''
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
        # self.alpha = 0.2
        self.alpha = nn.Parameter(torch.FloatTensor([0.2]))  # alpha 作为标量参数进行初始化


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
        # 第一种计算
        # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
        graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
        e = self.leaky_relu(graph.edata.pop('e'))

        # compute softmax
        graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))

        w = graph.edata['w']
        graph.edata['w'] = edge_softmax(graph, w)
        # print(w)
        # exit(0)
        graph.edata['a'] = graph.edata['a'].squeeze(1).squeeze(1)
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
        self.eps = 0.1

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

#尝试接入LLM：
from transformers import AutoTokenizer, AutoModel

class LLM(torch.nn.Module):
    def __init__(self,  out_channels, llm_output_dim=768, model_name='bert-base-uncased'):#bert-base-uncased
        super(LLM, self).__init__()
        # self.conv1 = GCNConv(in_channels + llm_output_dim, hidden_channels)  # 更新输入特征维度
        # self.conv2 = GCNConv(hidden_channels, out_channels)

        # self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        # self.llm = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        model_name_or_path = '/home/yyj/MDNN-AC/AutoAC-main/LLM'
        # # 初始化 LLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.llm = AutoModel.from_pretrained(model_name_or_path).to(device)
        # 打印模型输出的维度
        # print("Output hidden size:", self.llm.config.hidden_size)  # 输出特征维度

        #接一层线性层
        self.fc1 = nn.Linear(llm_output_dim, out_channels).to(device)


    def forward(self, x):
        # 生成节点描述并通过 LLM 获取嵌入
        node_texts = [f"Node feature values: {','.join(map(str, feature.tolist()))}" for feature in x]
        inputs = self.tokenizer(node_texts, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            llm_outputs = self.llm(**inputs)
            llm_embeddings = llm_outputs.last_hidden_state.mean(dim=1)
        #下接的线性层
        x = self.fc1(llm_embeddings)

        # # 将 LLM 输出的嵌入与原始节点属性 x 结合
        # x = torch.cat([x, llm_embeddings], dim=-1)  # 这里 x 的最后一维将是 in_channels + llm_output_dim
        #
        # # 通过 GNN 层处理图结构
        # x = self.conv1(x, edge_index).relu()
        # x = self.conv2(x, edge_index)
        return x