import torch

def init_delta(size, epsilon=1e-1, init_type="zero"):
    if init_type=="zero":
        delta = torch.zeros(size)
    elif init_type=="rand": # 随机分布
        delta = torch.rand(size) * epsilon
    elif init_type=="randn": # 正态分布
        delta = torch.randn(size) * epsilon
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