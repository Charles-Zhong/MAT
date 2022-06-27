import math
import torch

def init_delta(size, epsilon=1e-1, init_type="zero"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if init_type=="zero": # 全零
        delta = torch.zeros(size, device=device)
    elif init_type=="rand": # 随机分布
        delta = torch.rand(size, device=device) * epsilon
    elif init_type=="randn": # 正态分布
        delta = torch.randn(size, device=device) * epsilon
    return delta

def update_delta(delta, Adv_epsilon, Adv_max_norm):
    delta_grad = delta.grad # delta的梯度
    delta_grad_norm = torch.norm(delta_grad.view(delta_grad.size(0), -1), p=2, dim=1).view(-1, 1, 1) # .view：摊平  .norm：求L2范数  .view：重新设置shape
    delta_grad_norm = torch.clamp(delta_grad_norm, min=1e-8)  # 设置最小值，避免下一步除数为0
    new_delta = (delta + Adv_epsilon * delta_grad / delta_grad_norm)  # 新的扰动 δ = δ + ε * g/||g||
    if Adv_max_norm > 0:  # 限制扰动L2大小  Adv_max_norm == 0 则不限制
        delta_norm = torch.norm(new_delta.view(new_delta.size(0), -1), p=2, dim=1)
        exceed_mask = (delta_norm > Adv_max_norm).to(delta_norm)
        if(sum(exceed_mask) != 0):  # 存在超出限制的扰动大小
            reweights = (Adv_max_norm / delta_norm * exceed_mask + (1-exceed_mask)).view(-1, 1, 1) # 缩减比例
            new_delta = (new_delta * reweights)  # 按比例缩减到norm-ball内
    return new_delta

def ls(P, Q, task_name):
    task_type = "classification" if task_name != "STS-B" else "regression"
    if(task_type == "classification"):
        ls_fn = torch.nn.KLDivLoss(reduction='batchmean')
        return ls_fn(P.softmax(dim=-1).log(), Q.softmax(dim=-1)) + ls_fn(Q.softmax(dim=-1).log(), P.softmax(dim=-1))
    elif(task_type == "regression"):
        ls_fn = torch.nn.MSELoss(reduction="sum")
        return ls_fn(P, Q)

def SGLD(x, grad, step, epsilon):
    noise = init_delta(x.size(), epsilon=epsilon, init_type="randn")
    x = x - step * grad + math.sqrt(2 * step) * noise
    return x