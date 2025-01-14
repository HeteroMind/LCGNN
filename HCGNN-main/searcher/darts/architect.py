import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

from utils.tools import *

def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])

class Architect_Darts(object):
    
    def __init__(self, model, args):
        self.args = args
        self.network_momentum = args.network_momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
            lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
    '''
    是一个方法 _compute_unrolled_model，用于计算模型的"展开"（unrolled）版本，这在优化神经网络架构中很常见。以下是该方法的主要部分：

    def _compute_unrolled_model(self, X, y, eta, network_optimizer)：这是方法的定义，
    接受四个参数：X（输入数据），y（目标数据），eta（学习率），以及 network_optimizer（网络优化器，用于更新神经网络参数）。
    
    loss = self.model._loss(X, y, is_valid=False)：这一行代码计算训练损失（train loss），self.model._loss 方法用于计算损失函数的值，
    传入输入数据 X 和目标数据 y，并且 is_valid 参数可能用于控制是否进行有效性检查。
    
    theta = _concat(self.model.parameters()).data：这一行代码将神经网络模型 self.model 的参数连接成一个一维张量 theta，以便进行后续的计算。
    
    try ... except ...：这是一个异常处理块，用于尝试获取网络优化器中存储的动量（momentum）信息，以用于后续计算。如果动量信息不可用，将使用零向量作为 moment。
    
    dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay*theta：这一行代码计算梯度 dtheta，它包括两部分：
    
    第一部分使用 torch.autograd.grad 计算损失 loss 相对于神经网络参数的梯度。
    第二部分加上了L2正则化（self.network_weight_decay），以控制参数的大小。
    unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))：这一行代码构建了一个"展开"模型 unrolled_model，
    该模型是通过在 theta 上执行一步更新（theta.sub(eta, moment+dtheta)）得到的。这种展开操作通常在神经结构搜索中用于计算梯度。
    
    最终，unrolled_model 可以用于执行后续的优化或计算，通常用于搜索神经网络的架构或超参数。展开操作是NAS中的一种关键技术，用于评估不同架构的性能。'''
    def _compute_unrolled_model(self, X, y, eta, network_optimizer):
        loss = self.model._loss(X, y, is_valid=False) #train loss
        theta = _concat(self.model.parameters()).data# w
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        # print(f"self.model.parameters: {self.model.parameters()}")
        dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay*theta#gradient, L2 norm
        unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta)) # one-step update, get w' for Eq.7 in the paper
        return unrolled_model

    def step(self, X, y, eta, network_optimizer, unrolled):
        self.optimizer.zero_grad()
        if unrolled:
            self._backward_step_unrolled(X, y, eta, network_optimizer)
        else:
            self._backward_step(X, y, is_valid=True)
        self.optimizer.step()

    def _backward_step(self, X, y, is_valid=True):
        loss = self.model._loss(X, y, is_valid)
        loss.backward()

    def _backward_step_unrolled(self, X, y, eta, network_optimizer):
        unrolled_model = self._compute_unrolled_model(X, y, eta, network_optimizer)
        unrolled_loss = unrolled_model._loss(X, y, is_valid=True) # validation loss
        unrolled_loss.backward() # one-step update for w?
        dalpha = [v.grad for v in unrolled_model.arch_parameters()] #L_vali w.r.t alpha
        vector = [v.grad.data for v in unrolled_model.parameters()] # gradient, L_train w.r.t w, double check the model construction
        implicit_grads = self._hessian_vector_product(vector, X, y)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        #update alpha, which is the ultimate goal of this func, also the goal of the second-order darts
        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset+v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, X, y, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v) # R * d(L_val/w', i.e., get w^+
        loss = self.model._loss(X, y, is_valid=False) # train loss
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters()) # d(L_train)/d_alpha, w^+

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2 * R, v) # get w^-, need to subtract 2 * R since it has add R
        loss = self.model._loss(X, y, is_valid=False)# train loss
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())# d(L_train)/d_alpha, w^-

        #reset to the orignial w, always using the self.model, i.e., the original model
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
