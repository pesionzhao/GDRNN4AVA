"""
@File :Zoeppritz.py 
@Author :LiangXingcheng fixed by Pesion
@Date :2023/9/19
@Desc : 
"""
import torch
import numpy as np
from Forward.Zoeppritz_Complex import ZoeppritzComp


# TODO 待优化方向 1, 每一次调用Forward都要初始化一次Zoeppritz(), 2 梯度回传用的是单角度或多角度

# 由于结果可能有虚数, 所以正传无法脱离numpy

# 自定义Zoeppritz模型正传与反传
class MyZoeppritzOneTheta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vp, vs, rho, theta, wavemat, input_len, ntraces):
        device = vp.device
        ctx.zoeforward = ZoeppritzComp(ntraces, input_len)
        vp, vs, rho = vp.cpu().detach().numpy(), vs.cpu().detach().numpy(), rho.cpu().detach().numpy()

        # 为使变量在backward中可用，save_for_backward只能保存Variable/Tensor，其他类型使用ctx.xyz = xyz形式保存
        ctx.vp = vp
        ctx.vs = vs
        ctx.rho = rho
        ctx.theta = theta
        ctx.input_len = input_len
        ctx.theta_len = np.size(theta, 0)
        ctx.wavemat = wavemat
        ctx.ntraces = ntraces
        ctx.device = device
        sys = torch.tensor(ctx.zoeforward.forward(vp, vs, rho, theta, ctx.wavemat), dtype=torch.float32).to(device)

        sys = torch.sum(sys, dim=-1)  # 输出在角度方向上叠加, shape: shape:[batch, layers], 反传时的grad_output与其维度一致

        return sys

    # 这里的grad_out为链式法则链的上一层梯度
    @staticmethod
    def backward(ctx, grad_output):  # grad_output.shape = [batchsize, layers, (ntheta)]由于forward做了角度叠加,所以没有ntheta
        # import pdb
        # pdb.set_trace()
        # 雅可比矩阵
        G_mat = ctx.zoeforward.jacobian(ctx.vp, ctx.vs, ctx.rho, np.array([ctx.theta[-1]]),
                                        ctx.wavemat)  # shape:[batch, layers, 3*layers]
        G_mat = torch.tensor(G_mat, dtype=torch.float32, device=ctx.device)
        # grad_output = grad_output.cpu().detach().numpy()  # shape:[batch, layers, (ntheta)]这里没有theta

        # 和之前的梯度进行相乘以符合链式法则
        if ctx.ntraces != 1:
            grad_output = grad_output.unsqueeze(-2)  # grad_output.shape = [batchsize, 1, layers]
            delta_model = torch.matmul(grad_output, G_mat)  # [batch, 1, 3*layers]
            delta_model = delta_model.squeeze(-2)  # [batch, 1, 3*layers]
            # print(torch.max(torch.abs(delta_model), dim=-1))
            # delta_model = delta_model / torch.max(torch.abs(delta_model))  # TODO 这个处理？？

        else:
            delta_model = torch.matmul(grad_output, G_mat)  # [layers, 3*layers]@[layers,] = [3*layers]
            # delta_model = delta_model / torch.max(torch.abs(delta_model[:, ]))

        # import pdb
        # pdb.set_trace()
        alpha = 0.1

        return alpha * delta_model[..., 0: ctx.input_len].T, \
               alpha * delta_model[..., ctx.input_len: 2 * ctx.input_len].T, \
               alpha*0.1 * delta_model[..., 2 * ctx.input_len: 3 * ctx.input_len].T, \
               None, None, None, None


class MyZoeppritzMultiTheta(torch.autograd.Function):
    ## Version 1
    @staticmethod
    def forward(ctx, vp, vs, rho, theta, wavemat, input_len, ntraces):
        device = vp.device
        ctx.zoeforward = ZoeppritzComp(ntraces, input_len)
        vp, vs, rho = vp.cpu().detach().numpy(), vs.cpu().detach().numpy(), rho.cpu().detach().numpy()

        # 为使变量在backward中可用，save_for_backward只能保存Variable/Tensor，其他类型使用ctx.xyz = xyz形式保存
        ctx.vp = vp
        ctx.vs = vs
        ctx.rho = rho
        ctx.theta = theta
        ctx.input_len = input_len
        ctx.theta_len = np.size(theta, 0)
        ctx.wavemat = wavemat
        ctx.ntraces = ntraces
        ctx.device = device

        sys = torch.tensor(ctx.zoeforward.forward(vp, vs, rho, theta, ctx.wavemat), dtype=torch.float32).to(device)

        sys = torch.sum(sys, dim=-1)  # 输出在角度方向上叠加, shape: shape:[batch, layers], 反传时的grad_output与其维度一致

        return sys

    # 这里的grad_out为链式法则链的上一层梯度
    @staticmethod
    def backward(ctx, grad_output):  # grad_output.shape = [batchsize, layers, (ntheta)] 由于forward做了角度叠加,所以没有ntheta
        # import pdb
        # pdb.set_trace()
        G_mat = ctx.zoeforward.jacobian(ctx.vp, ctx.vs, ctx.rho, ctx.theta,
                                        ctx.wavemat)  # shape:[batch, layers*ntheta, 3*layers]

        grad_output = grad_output.cpu().detach().numpy()  # shape:[batch, layers, (ntheta)]这里没有theta
        # import pdb
        # pdb.set_trace()

        if ctx.ntraces != 1:
            delta_model = np.matmul(grad_output[:, np.newaxis, :], G_mat)  # [batch, 1, 3*layers]
            delta_model = delta_model / np.max(np.abs(delta_model), axis=-1)  # TODO 这个处理？？

        else:
            delta_model = np.matmul(grad_output, G_mat)
            delta_model = delta_model / np.max(np.abs(delta_model[:, ]))
        # import pdb
        # pdb.set_trace()

        alpha = 10 / 1000

        if ctx.ntraces != 1:
            return torch.tensor(alpha * delta_model[:, 0, 0: ctx.input_len].T).to(ctx.device), \
                torch.tensor(alpha * delta_model[:, 0, ctx.input_len: 2 * ctx.input_len].T).to(ctx.device), \
                torch.tensor(0.1 * alpha * delta_model[:, 0, 2 * ctx.input_len: 3 * ctx.input_len].T).to(ctx.device), \
                None, None, None, None
        else:
            return torch.tensor(alpha * delta_model[0: ctx.input_len]).to(ctx.device), \
                torch.tensor(alpha * delta_model[ctx.input_len: 2 * ctx.input_len]).to(ctx.device), \
                torch.tensor(0.1 * alpha * delta_model[2 * ctx.input_len: 3 * ctx.input_len]).to(ctx.device), \
                None, None, None, None
