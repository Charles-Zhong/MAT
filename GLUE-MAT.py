import os
import csv
import time
import torch
from utils import preprocess, function, args
from tqdm.auto import tqdm

args = args.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

run_time = str(int(time.time()))
log_path = "logs/" + run_time
os.makedirs(log_path)
file = open(log_path + "/" + run_time +".log","w") # 设置日志文件

# 加载模型训练集
local_model_path = "models/" # "models/" 预先下载到本地的模型文件夹, 为""时自动从huggingface下载模型。
model, dataloader, metric= preprocess.preprocess(args.task_name, local_model_path + args.model_name, args.batch_size, args.seed)
train_dataloader,eval_dataloader,test_dataloader = dataloader
iterations = args.epochs * len(train_dataloader)

# Training
print(time.ctime(), file=file)
print("*"*20, "Training", "*"*20, file=file)  # 训练任务
print("TASK:", args.task_name, file=file)
print("MODEL:", args.model_name, file=file)
print("DEVICE:", device, file=file)
print("="*16, "General Training", "="*16, file=file)  # 常规训练参数
print("EPOCH_NUM:", args.epochs, file=file)
print("BATCH_SIZE:", args.batch_size, file=file)
print("="*18, "MAT Training", "="*18, file=file)  # MAT训练参数
print("Adversarial_Training_type:", "MAT", file=file)
print("Adversarial_init_epsilon:", args.adv_init_epsilon, file=file)
print("Adversarial_init_type:", args.adv_init_type, file=file)
print("Sampling_times_theta:", args.sampling_times_theta, file=file)
print("Sampling_times_delta:", args.sampling_times_delta, file=file)
print("Sampling_noise_theta:", args.sampling_noise_theta, file=file)
print("Sampling_noise_delta:", args.sampling_noise_delta, file=file)
print("Sampling_step_theta:", args.sampling_step_theta, file=file)
print("Sampling_step_delta:", args.sampling_step_delta, file=file)
print("lambda:", args.lambda_s, file=file)
print("beta_s:", args.beta_s, file=file)
print("beta_p:", args.beta_p, file=file)
print("*"*50, file=file)
model.to(device)
progress_bar = tqdm(range(iterations))
progress_bar.set_description("Training...")
eval_metric_list = []
score_list = []
ite = 0
for i in range(args.epochs):

    ###################  Train-begin  ###################
    model.train()
    print("-"*20, "EPOCH:", i, "-"*20, file=file)
    print("Training...", end='', file=file)

    for batch in train_dataloader:
        batch = {key: batch[key].to(device) for key in batch}

        # [begin] MAT Training
        # 1.init delta & inputs
        ## 1.1 初始化模型输入inputs
        if "bert-" in args.model_name:  # bert模型输入inputs: "attention_mask","labels","token_type_ids",「inputs_embeds」和「input_ids」参数二选一
            inputs = {"attention_mask": batch["attention_mask"],"labels": batch["labels"], "token_type_ids": batch["token_type_ids"]}
        elif "roberta-" in args.model_name:  # roberta模型输入inputs: "attention_mask","labels",「inputs_embeds」和「input_ids」参数二选一
            inputs = {"attention_mask": batch["attention_mask"], "labels": batch["labels"]}

        ## 1.2 获得batch的word_embedding
        if "bert-" in args.model_name:
            word_embedding = model.bert.embeddings.word_embeddings(batch["input_ids"])
        elif "roberta-" in args.model_name:
            word_embedding = model.roberta.embeddings.word_embeddings(batch["input_ids"])
        
        ## 1.3 初始化[抽样扰动delta]和[扰动分布均值mean_delta]
        delta = function.init_delta(word_embedding.size(), args.adv_init_epsilon, args.adv_init_type)
        delta.requires_grad = True
        mean_delta = delta.clone().detach()  # 初始化delta的分布均值mean_delta

        ## 1.4 备份模型参数
        back_parameters = model.state_dict()
        mean_theta = model.state_dict()

        # 2.stochastic gradient langevin dynamics sampling
        ## 2.1 sampling perturbation (delta)
        for k in range(args.sampling_times_delta):
            ### 构造带有扰动的输入
            inputs["inputs_embeds"] = delta + word_embedding.detach()
            ### 前向传播
            output_normal = model(**batch)
            output_adv = model(**inputs)
            loss_adv = function.ls(output_normal.logits, output_adv.logits, args.task_name)
            ### 反向传播
            loss_adv.backward()
            ### SGLD采样
            delta.data = function.SGLD(delta.data, - delta.grad, args.sampling_step_delta, args.sampling_noise_delta)
            delta.grad.zero_()
            ### 更新扰动的分布均值
            mean_delta.data = args.beta_s * mean_delta.data + (1 - args.beta_s) * delta.data

        ## 2.2 sampling model parameters (theta)
        for k in range(args.sampling_times_theta):
            ### 清空模型参数的梯度
            for p in model.parameters():
                if p.grad!=None:
                    p.grad.zero_()
            ### 构造带有扰动的输入
            if "bert-" in args.model_name: # 每次backward()会丢失word_embedding的计算图，因此需要每次计算一遍word_embedding
                word_embedding = model.bert.embeddings.word_embeddings(batch["input_ids"])
            elif "roberta-" in args.model_name:
                word_embedding = model.roberta.embeddings.word_embeddings(batch["input_ids"])
            inputs["inputs_embeds"] = mean_delta.detach() + word_embedding
            ### 前向传播
            output_normal = model(**batch)
            output_adv = model(**inputs)
            loss_sum = output_normal.loss + args.lambda_s * function.ls(output_normal.logits, output_adv.logits, args.task_name)
            ### 反向传播
            loss_sum.backward()
            ### SGLD采样并更新分布均值
            for name, p in model.named_parameters():
                p.data = function.SGLD(p.data, p.grad, args.sampling_step_theta * (iterations-ite)/iterations, args.sampling_noise_theta) # 将模型参数更新为新的采样
                mean_theta[name] = args.beta_s * mean_theta[name] + (1 - args.beta_s) * p.data # 更新模型参数的分布均值 
                          
        # 3.update model parameters
        for key in back_parameters:
            back_parameters[key] = args.beta_p * back_parameters[key] + (1 - args.beta_p) * mean_theta[key] # 调整备份的模型参数
        model.load_state_dict(back_parameters) # 将模型参数更新为这次迭代的模型参数
        # [end] MAT Training
        ite += 1
        progress_bar.update(1)
    ###################  Train-end  ###################  

    ###################  Validate-begin  ###################
    model.eval()
    print("\rEvaling...", end='', file=file)
    for batch in eval_dataloader:
        batch = {key: batch[key].to(device) for key in batch}
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1) if args.task_name != "STS-B" else outputs.logits.squeeze()
        metric.add_batch(predictions=predictions, references=batch["labels"])
    metric_data = metric.compute()
    eval_metric_list.append(metric_data)
    score = list(metric_data.values())[0]
    score_list.append(score)
    print("\rMetric:", metric_data, file=file)
    print("-"*50, file=file)
    ###################  Validate-end  ###################

    ###################  Test-begin  ###################
    if score==max(score_list):
        with(open(log_path + "/" + args.task_name+".tsv","w")) as f:
            tsv_writer = csv.writer(f, delimiter='\t')
            tsv_writer.writerow(["index","prediction"])
            for t,batch in enumerate(test_dataloader):
                batch = {key: batch[key].to(device) for key in batch}
                with torch.no_grad():
                    logits=model(**batch).logits
                predict_batch = logits.argmax(1)
                for id, pre in enumerate(predict_batch):
                    predict_label = preprocess.TASKS_TO_LABEL[args.task_name][pre.item()]
                    tsv_writer.writerow([id + t * args.batch_size, predict_label])
            f.close()
    ###################  Test-end  ###################

# Best score in validation set
print("*"*19, "Best Score", "*"*19, file=file)
print("EPOCH:", score_list.index(max(score_list)), file=file)
print("Best Metric:", eval_metric_list[score_list.index(max(score_list))], file=file)
print("*"*50, file=file)

file.close()
