import time
import torch
import numpy as np
import copy
import gc
import os
from torch.utils.tensorboard import SummaryWriter

from utils.tools import *
from models.model_manager import ModelManager

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
class Retrainer2:
    def __init__(self, new_features_list,data_info, idx_info, train_info,writer, args):
        self.args = args
        self._logger = args.logger
        # 从 data_info 中获取数据和图信息
        _, self.labels, self.g, self.type_mask, self.dl, self.in_dims, self.num_classes = data_info
        self.train_idx, self.val_idx, self.test_idx = idx_info
        self.criterion = train_info

        self._writer = writer
        self.save_path = 'checkpoint/checkpoint_retrain_{}.pt'.format(args.time_line)
        # temp = genotype_dir_name.split('_')
        # temp = temp[:-1]
        # _genotype_dir_name = '_'.join(temp)
        # self.genotype_dir_name = _genotype_dir_name
        self.features_list=new_features_list
        # 将数据和标签转换为 PyTorch 张量
        self.input, self.target = convert_np2torch(self.features_list, self.labels, args)
        self.combined_features = torch.cat(self.input, dim=0).to(device)#当启用DP_AC_2.PY时，这句话要启用
    def _is_save(self, train_loss, val_loss):
        if val_loss < self._bst_val_loss:
            self._bst_val_loss = val_loss
            return True
        return False

    def _save_search_info(self, model):
        torch.save(model.state_dict(), self.save_path)

    def retrain2(self, fixed_model, cur_repeat):
    #def retrain1(self, fixed_model,onehot_feature_list,node_type_feature, features_list,type_mask,cur_repeat):#这是用于残差注意力网络
        model = fixed_model.to(device)#.cuda()
        ##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!这是后加的
        #checkpoint = torch.load('D:\\pycharm_item\\MDNN-AC\\AutoAC-main\\checkpoint\\save\\net_params_{}.pt'.format(self.args.time_line))  # , map_location=torch.device('gpu'))
        #hgnn_model_state_dict = {k: v for k, v in checkpoint.items() if 'hgnn_model' in k}
        ## 加载状态字典
        #model.load_state_dict(hgnn_model_state_dict, strict=False)
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # 根据参数选择优化器和学习率调度器
        if self.args.useSGD:
            optimizer = torch.optim.SGD(
                fixed_model.parameters(),
                self.args.lr,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   float(self.args.retrain_epoch * self.args.inner_epoch),
                                                                   eta_min=self.args.lr_rate_min)
        elif self.args.use_adamw:
            optimizer = torch.optim.AdamW(fixed_model.parameters(), weight_decay=self.args.weight_decay)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=self.args.schedule_step_retrain,
                                                            max_lr=1e-3, pct_start=0.05)
            # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=self.args.schedule_step_retrain, max_lr=5e-4, pct_start=0.05)
        else:
            scheduler = None
            optimizer = torch.optim.Adam(fixed_model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        # 创建 checkpoint 目录
        if not os.path.exists('checkpoint/'):
            os.makedirs('checkpoint/')
        # 创建索引生成器
        train_idx_generator = index_generator(batch_size=self.args.batch_size, indices=self.train_idx)
        val_idx_generator = index_generator(batch_size=self.args.batch_size, indices=self.val_idx, shuffle=False)
        # 创建 EarlyStopping_Retrain 实例
        # earlystop = EarlyStopping_Retrain(logger=self._logger, patience=self.args.patience)
        earlystop = EarlyStopping_Retrain(logger=self._logger, patience=self.args.patience_retrain)
        # 初始化最佳验证集损失
        self._bst_val_loss = np.inf
        # 循环训练
        for epoch in range(self.args.retrain_epoch):
            t_start = time.time()

            if self.args.useSGD or self.args.use_adamw:
                # if self.args.useSGD:
                # scheduler.step()
                lr = scheduler.get_lr()[0]
            else:
                lr = optimizer.state_dict()['param_groups'][0]['lr']

            # train model   # 训练模型
            model.train()

            if self.args.use_minibatch is False:
                node_embedding, _, logits = model(self.combined_features)#node_embedding, _, logits = model(self.combined_features,self.data_info[2].edges())
                #node_embedding, _, logits = model(self.combined_features, self.data_info[2].edges())
                #_,logits,_,_=model(onehot_feature_list, node_type_feature, features_list, type_mask)#这是用于残差注意力网络

                logits_train = logits[self.train_idx].to(device)
                target = self.target[self.train_idx]
                train_loss = self.criterion(logits_train, target)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                if self.args.use_adamw:
                    scheduler.step(epoch + 1)
            else:
                # 使用 minibatch 训练
                minibatch_data_info = model.gnn_model_manager.get_graph_info()
                self.adjlists, self.edge_metapath_indices_list = minibatch_data_info

                train_loss_avg = 0

                for step in range(train_idx_generator.num_iterations()):
                    _t_start = time.time()

                    train_idx_batch = train_idx_generator.next()
                    train_idx_batch.sort()
                    train_g_list, train_indices_list, train_idx_batch_mapped_list = parse_minibatch(
                        self.adjlists, self.edge_metapath_indices_list, train_idx_batch, device,
                        self.args.neighbor_samples)

                    node_embedding, _, logits = model(self.combined_features, (
                    train_g_list, train_indices_list, train_idx_batch_mapped_list, train_idx_batch))

                    logits_train = logits.to(device)
                    train_loss = self.criterion(logits_train, self.target[train_idx_batch])

                    optimizer.zero_grad()
                    train_loss.backward()
                    optimizer.step()

                    self._logger.info('Epoch_batch_{:05d} | lr {:.4f} |Train_Loss {:.4f}| Time(s) {:.4f}'.format(
                        step, lr, train_loss.item(), time.time() - _t_start))

                    train_loss_avg += train_loss.item()

                train_loss_avg /= train_idx_generator.num_iterations()
                train_loss = train_loss_avg

            # infer model on validation set    # 在验证集上进行推断
            model.eval()
            with torch.no_grad():
                # input, target = convert_np2torch(self.features_list, self.labels, self.args, y_idx=self.val_idx)
                if self.args.use_minibatch is False:

                    _, _, logits = model(self.combined_features)
                    #_, _, logits = model(self.combined_features, self.data_info[2].edges())
                    #_, logits, _, _ = model(onehot_feature_list, node_type_feature, features_list,type_mask)  # 这是用于残差注意力网络
                    logits_val = logits[self.val_idx].to(device)
                else:
                    logits_val = []
                    val_idx_generator = index_generator(batch_size=self.args.batch_size, indices=self.val_idx,
                                                        shuffle=False)
                    for iteration in range(val_idx_generator.num_iterations()):
                        val_idx_batch = val_idx_generator.next()
                        val_g_list, val_indices_list, val_idx_batch_mapped_list = parse_minibatch(
                            self.adjlists, self.edge_metapath_indices_list, val_idx_batch, device,
                            self.args.neighbor_samples)
                        node_embedding, _, logits = model(self.combined_features, (
                        val_g_list, val_indices_list, val_idx_batch_mapped_list, val_idx_batch))
                        logits_val.append(logits)
                        # logits_val.append(logits[val_idx_batch])
                    logits_val = torch.cat(logits_val, 0).to(device)
                target = self.target[self.val_idx]
                val_loss = self.criterion(logits_val, target)

            t_end = time.time()
            self._logger.info('Epoch {:05d} | lr {:.5f} |Train_Loss {:.4f} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
                epoch, lr, train_loss, val_loss.item(), t_end - t_start))

            self._writer.add_scalar(f'Retrain_TrainLoss_{cur_repeat}', train_loss, global_step=epoch)
            self._writer.add_scalar(f'Retrain_ValLoss_{cur_repeat}', val_loss.item(), global_step=epoch)

            if self._is_save(train_loss, val_loss.item()):
                self._save_search_info(model)

            earlystop(train_loss, val_loss.item())
            if earlystop.early_stop:
                self._logger.info('Early stopping!')
                break

        return model

    def test2(self, model, cur_repeat):
    #def test1(self, model,onehot_feature_list,node_type_feature, features_list,type_mask, cur_repeat):
        self._logger.info('\ntesting...')
        # 加载保存的模型参数
        model.load_state_dict(torch.load('checkpoint/checkpoint_retrain_{}.pt'.format(self.args.time_line)))
        model.eval()
        # 创建提交目录
        # if not os.path.exists(f'submit/submit_{self.genotype_dir_name}_{self.args.time_line}'):
        #     os.makedirs(f'submit/submit_{self.genotype_dir_name}_{self.args.time_line}')

        if not os.path.exists(f'submit/submit_{save_dir_name(self.args)}'):
            os.makedirs(f'submit/submit_{save_dir_name(self.args)}')

        self._logger.info(f'submit dir: submit/submit_{save_dir_name(self.args)}')

        #self._logger.info(f'submit dir: submit/submit_{self.genotype_dir_name}')

        with torch.no_grad():
            if self.args.use_minibatch is False:
                _, logits, _= model(self.combined_features)
                #node_embedding, _, logits = model(self.combined_features, self.data_info[2].edges())
                #_, logits, _, _ = model(onehot_feature_list, node_type_feature, features_list, type_mask)  # 这是用于残差注意力网络
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
                    node_embedding, _, logits = model(self.combined_features, (
                    test_g_list, test_indices_list, test_idx_batch_mapped_list, test_idx_batch))
                    logits_test.append(logits)

                logits_test = torch.cat(logits_test, 0).to(device)

            if self.args.dataset == 'IMDB':
                pred = (logits_test.cpu().numpy() > 0).astype(int)
                # self.dl.gen_file_for_evaluate(test_idx=self.test_idx, label=pred,
                #                             file_path=(f'submit/submit_{self.genotype_dir_name}_{self.args.time_line}/{self.args.dataset}_{cur_repeat + 1}.txt'), mode='multi')
                #self.dl.gen_file_for_evaluate(test_idx=self.test_idx, label=pred,file_path=('submit/submit_{self.genotype_dir_name}/{self.args.dataset}_{cur_repeat + 1}.txt'),mode='multi')
                self.dl.gen_file_for_evaluate(test_idx=self.test_idx, label=pred,file_path=(f'submit/submit_{save_dir_name(self.args)}/{self.args.dataset}_{cur_repeat + 1}.txt'),mode='multi')

                self._logger.info(self.dl.evaluate(pred))
                print(
                    f"{bcolors.WARNING}Warning: If you want to obtain test score, please submit online on biendata.{bcolors.ENDC}")
            else:
                pred = logits_test.cpu().numpy().argmax(axis=1)
                # self.dl.gen_file_for_evaluate(test_idx=self.test_idx, label=pred,
                #                             file_path=(f'submit/submit_{self.genotype_dir_name}_{self.args.time_line}/{self.args.dataset}_{cur_repeat + 1}.txt'))
                #self.dl.gen_file_for_evaluate(test_idx=self.test_idx, label=pred,file_path=(f'submit/submit_{self.genotype_dir_name}/{self.args.dataset}_{cur_repeat + 1}.txt'))
                self.dl.gen_file_for_evaluate(test_idx=self.test_idx, label=pred, file_path=(f'submit/submit_{save_dir_name(self.args)}/{self.args.dataset}_{cur_repeat + 1}.txt'))

                onehot = np.eye(self.num_classes, dtype=np.int32)
                pred = onehot[pred]
                self._logger.info(self.dl.evaluate(pred))
                print(f"{bcolors.WARNING}Warning: If you want to obtain test score, please submit online on biendata.{bcolors.ENDC}")

