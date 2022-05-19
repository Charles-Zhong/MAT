import math
import time
import torch
import argparse
import preprocess
import perturbation
from torch.nn import MSELoss
import torch.nn.functional as F
from tqdm.auto import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on GLUE with MAT training mode.")
    parser.add_argument("--task_name", type=str, default="SST-2", choices=["CoLA", "SST-2", "MRPC", "STS-B", "QQP", "MNLI-m", "MNLI-mm", "QNLI", "RTE", "WNLI"], help="The name of the glue task to train on.")
    parser.add_argument("--model_name", type=str, default="models/bert-base-uncased", choices=["models/bert-base-uncased", "models/roberta-base"],help="Finetune base model.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for the training dataloader.")
    parser.add_argument("--epochs", type=int, default=5, help="Total number of training epochs to perform.")
    parser.add_argument("--adv_init_epsilon", type=float, default=1e-2, help="Initialization size of adversarial perturbation.")
    parser.add_argument("--adv_init_type", type=str, default="zero", help="Initialization type of adversarial perturbation.")
    parser.add_argument("--sampling_times_theta", type=int, default=20, help="Stochastic gradient langevin dynamics sampling times for model parameters.")
    parser.add_argument("--sampling_times_delta", type=int, default=3, help="Stochastic gradient langevin dynamics sampling times for adversarial perturbation.")
    parser.add_argument("--sampling_noise_theta", type=float, default=0, help="Stochastic gradient langevin dynamics sampling noise for model parameters.")
    parser.add_argument("--sampling_noise_delta", type=float, default=0, help="Stochastic gradient langevin dynamics sampling noise for adversarial perturbation.")
    parser.add_argument("--sampling_step_theta", type=float, default=3e-5, help="Stochastic gradient langevin dynamics sampling step for model parameters.")
    parser.add_argument("--sampling_step_delta", type=float, default=1e-3, help="Stochastic gradient langevin dynamics sampling step for adversarial perturbation.")
    parser.add_argument("--lambda_s", type=float, default=1, help="Tuning parameter lambda of the objective function.")
    parser.add_argument("--beta", type=float, default=0.1, help="Exponential damping beta for stability in Stochastic gradient langevin dynamics sampling and parameter updating.")
    args = parser.parse_args()
    return args

args = parse_args()

# 设置任务和模型
task_name = args.task_name # ["CoLA", "SST-2", "MRPC", "STS-B", "QQP", "MNLI-m", "MNLI-mm", "QNLI", "RTE", "WNLI"]
model_name = args.model_name # ["bert-base-uncased", "roberta-base"]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 设置超参数
# 训练参数
seed = args.seed
batch_size = args.batch_size
epochs = args.epochs
# 对抗训练参数
adv_init_epsilon = args.adv_init_epsilon # 初始化扰动
adv_init_type = args.adv_init_type # ["zero","rand","randn"]
sampling_times_theta = args.sampling_times_theta # theta采样次数
sampling_times_delta = args.sampling_times_delta # delta采样次数
sampling_noise_theta = args.sampling_noise_theta # theta采样噪声
sampling_noise_delta = args.sampling_noise_delta # delta采样噪声
sampling_step_theta = args.sampling_step_theta # theta采样步长
sampling_step_delta = args.sampling_step_delta # theta采样步长
lambda_s = args.lambda_s # lambda λ
beta = args.beta # beta β

# 加载模型训练集
model, dataloader, metric= preprocess.preprocess(task_name, model_name, batch_size, seed)
train_dataloader,eval_dataloader,test_dataloader = dataloader

def ls(P, Q):
    task_type = "classification" if task_name != "STS-B" else "regression"
    if(task_type == "classification"):
        return F.kl_div(P.softmax(dim=-1).log(), Q.softmax(dim=-1), reduction='batchmean') + F.kl_div(Q.softmax(dim=-1).log(), P.softmax(dim=-1), reduction='batchmean')
    elif(task_type == "regression"):
        return MSELoss(P, Q, reduction="sum")

def SGLD(z, grad, step, epsilon):
    noise = perturbation.init_delta(z.size(), epsilon=epsilon, init_type="randn")
    z = z - step * grad + math.sqrt(2 * step) * noise
    return z

# all is ready!

train_start = time.time()
file = open("log/run: "+ str(int(train_start)) +".log","w") # 设置日志文件

# Training
print(time.ctime(), file=file)
print("*"*20, "Training", "*"*20, file=file)  # 训练任务
print("TASK:", task_name, file=file)
print("MODEL:", model_name, file=file)
print("DEVICE:", device, file=file)
print("="*16, "General Training", "="*16, file=file)  # 常规训练参数
print("EPOCH_NUM:", epochs, file=file)
print("BATCH_SIZE:", batch_size, file=file)
print("="*18, "MAT Training", "="*18, file=file)  # MAT训练参数
print("Adversarial_Training_type:", "MAT", file=file)
print("Adversarial_init_epsilon:", adv_init_epsilon, file=file)
print("Adversarial_init_type:", adv_init_type, file=file)
print("Sampling_times_theta:", sampling_times_theta, file=file)
print("Sampling_times_delta:", sampling_times_delta, file=file)
print("Sampling_noise_theta:", sampling_noise_theta, file=file)
print("Sampling_noise_delta:", sampling_noise_delta, file=file)
print("Sampling_step_theta:", sampling_step_theta, file=file)
print("Sampling_step_delta:", sampling_step_delta, file=file)
print("lambda:", lambda_s, file=file)
print("beta:", beta, file=file)
print("*"*50, file=file)
model.to(device)
progress_bar = tqdm(range(epochs * len(train_dataloader)))
progress_bar.set_description("Training...")
eval_metric_list = []
for i in range(epochs):
    print("-"*20, "EPOCH:", i, "-"*20, file=file)
    print("Training...", end='', file=file)
    model.train()

    for batch in train_dataloader:
        batch = {key: batch[key].to(device) for key in batch}

        # [begin] MAT Training
        # 1.init delta & inputs
        ## 1.1 初始化模型输入inputs
        if "bert-" in model_name:  # bert模型输入inputs: "attention_mask","labels","token_type_ids",「inputs_embeds」和「input_ids」参数二选一
            inputs = {"attention_mask": batch["attention_mask"],"labels": batch["labels"], "token_type_ids": batch["token_type_ids"]}
        elif "roberta-" in model_name:  # roberta模型输入inputs: "attention_mask","labels",「inputs_embeds」和「input_ids」参数二选一
            inputs = {"attention_mask": batch["attention_mask"], "labels": batch["labels"]}

        ## 1.2 获得batch的word_embedding
        if "bert-" in model_name:
            word_embedding = model.bert.embeddings.word_embeddings(batch["input_ids"])
        elif "roberta-" in model_name:
            word_embedding = model.roberta.embeddings.word_embeddings(batch["input_ids"])
        
        ## 1.3 初始化[抽样扰动delta]和[扰动分布均值mean_delta]
        delta = perturbation.init_delta(word_embedding.size(), adv_init_epsilon, adv_init_type)
        delta.requires_grad = True
        mean_delta = delta.clone().detach()  # 初始化delta的分布均值mean_delta

        ## 1.4 备份模型参数
        back_parameters = model.state_dict()
        mean_theta = model.state_dict()

        # 2.stochastic gradient langevin dynamics sampling
        ## 2.1 sampling perturbation (delta)
        for k in range(sampling_times_delta):
            ### 构造带有扰动的输入
            inputs["inputs_embeds"] = delta + word_embedding.detach()
            ### 前向传播
            loss_adv = ls(model(**inputs).logits, model(**batch).logits)
            ### 反向传播
            loss_adv.backward()
            ### SGLD采样
            delta.data = SGLD(delta.data, - delta.grad, sampling_step_delta, sampling_noise_delta)
            delta.grad.zero_()
            ### 更新扰动的分布均值
            mean_delta.data = beta * mean_delta.data + (1 - beta) * delta.data

        ## 2.2 sampling model parameters (theta)
        for k in range(sampling_times_theta):
            ### 清空模型参数的梯度
            for p in model.parameters():
                if p.grad!=None:
                    p.grad.zero_()
            ### 构造带有扰动的输入
            if "bert-" in model_name: # 每次backward()会丢失word_embedding的计算图，因此需要每次计算一遍word_embedding
                word_embedding = model.bert.embeddings.word_embeddings(batch["input_ids"])
            elif "roberta-" in model_name:
                word_embedding = model.roberta.embeddings.word_embeddings(batch["input_ids"])
            inputs["inputs_embeds"] = mean_delta.detach() + word_embedding
            ### 前向传播
            loss_sum = model(**batch).loss + lambda_s * ls(model(**inputs).logits, model(**batch).logits)
            ### 反向传播
            loss_sum.backward()
            ### SGLD采样并更新分布均值
            for name, p in model.named_parameters():
                p.data = SGLD(p.data, p.grad, sampling_step_theta, sampling_noise_theta) # 将模型参数更新为新的采样
                mean_theta[name] = beta * mean_theta[name] + (1 - beta) * p.data # 更新模型参数的分布均值 
                          
        # 3.update model parameters
        for key in back_parameters:
            back_parameters[key] = beta * back_parameters[key] + (1 - beta) * mean_theta[key] # 调整备份的模型参数
        model.load_state_dict(back_parameters) # 将模型参数更新为这次迭代的模型参数
        # [end] MAT Training
        progress_bar.update(1)

    print("\rEvaling...", end='', file=file)
    model.eval()
    for batch in eval_dataloader:
        batch = {key: batch[key].to(device) for key in batch}
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1) if task_name != "STS-B" else outputs.logits.squeeze()
        metric.add_batch(predictions=predictions, references=batch["labels"])
    score = metric.compute()
    eval_metric_list.append(score)
    print("\rMetric:", score, file=file)
    print("-"*50, file=file)

# Best score in eval
score_list = []
for m in eval_metric_list:
    score_list.append(list(m.values())[0])
print("*"*19, "Best Score", "*"*19, file=file)
print("EPOCH:", score_list.index(max(score_list)), file=file)
print("Metric:", eval_metric_list[score_list.index(max(score_list))], file=file)
print("*"*50, file=file)

file.close()
