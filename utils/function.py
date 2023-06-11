import math
import torch

def init_delta(size, epsilon=1e-1, init_type="zero"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if init_type == "zero":
        return torch.zeros(size, device=device)
    elif init_type == "rand":
        return torch.rand(size, device=device) * epsilon
    elif init_type == "randn":
        return torch.randn(size, device=device) * epsilon

def update_delta(delta, adv_epsilon, adv_max_norm):
    delta_grad = delta.grad
    delta_grad_norm = torch.norm(delta_grad.view(delta_grad.size(0), -1), p=float('inf'), dim=1).view(-1, 1, 1)
    delta_grad_norm = torch.clamp(delta_grad_norm, min=1e-8)
    new_delta = (delta + adv_epsilon * delta_grad / delta_grad_norm)
    if adv_max_norm > 0:
        delta_norm = torch.norm(new_delta.view(new_delta.size(0), -1), p=float('inf'), dim=1)
        exceed_mask = (delta_norm > adv_max_norm).to(delta_norm)
        if(sum(exceed_mask) != 0):
            reweights = (adv_max_norm / delta_norm * exceed_mask + (1-exceed_mask)).view(-1, 1, 1)
            new_delta = (new_delta * reweights)
    return new_delta

def ls(P, Q, task_name):
    task_type = "classification" if task_name != "STS-B" else "regression"
    if(task_type == "classification"):
        ls_fn = torch.nn.KLDivLoss(reduction='batchmean')
        return ls_fn(P.softmax(dim=-1).log(), Q.softmax(dim=-1)) + ls_fn(Q.softmax(dim=-1).log(), P.softmax(dim=-1))
    elif(task_type == "regression"):
        ls_fn = torch.nn.MSELoss(reduction="sum")
        return ls_fn(P, Q)

def SGLD(x, step, epsilon):
    noise = init_delta(x.size(), epsilon=epsilon, init_type="randn")
    x = x + math.sqrt(2 * step) * noise
    return x
