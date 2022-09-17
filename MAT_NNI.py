import os
import csv
import nni
import time
import torch
import random
import numpy as np
from tqdm.auto import tqdm
from utils import args, function, preprocess

args = nni.get_next_parameter()
random.seed(args["seed"])
np.random.seed(args["seed"])
torch.manual_seed(args["seed"])
torch.cuda.manual_seed_all(args["seed"])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

run_time = str(int(time.time()))
log_path = "logs/" + args["task_name"] + "/" + args["model_name"] + "/" + run_time
os.makedirs(log_path)
file = open(log_path + "/" + args["task_name"] + "_" + args["model_name"] + "_" + run_time + ".log", "w")  # 设置日志文件

# 加载模型训练集
local_model_path = args["model_path"]  + "/" if args["model_path"] != "" else "" # local_model_dir, 设置为""时会自动从huggingface下载model
model, dataloader, metric = preprocess.preprocess(args["task_name"], local_model_path + args["model_name"], args["batch_size"])
train_dataloader, eval_dataloader, test_dataloader = dataloader
model.to(device)

# 设置进度条
total_iterations = args["epochs"] * len(train_dataloader)
progress_bar = tqdm(range(total_iterations))

eval_step = len(train_dataloader) // args["eval_times"] # 默认1个Epoch评估10次

eval_metric_list = []  # 验证集metric列表
eval_score_list = []  # 验证集score列表
current_iteration = 0

# Training
print(time.ctime(), file=file)
print("*"*20, "Training", "*"*20, file=file)  # 训练任务
print("TASK:", args["task_name"], file=file)
print("MODEL:", args["model_name"], file=file)
print("="*16, "General Training", "="*16, file=file)  # 常规训练参数
print("SEED:", args["seed"], file=file)
print("EPOCH_NUM:", args["epochs"], file=file)
print("BATCH_SIZE:", args["batch_size"], file=file)
print("="*18, "MAT Training", "="*18, file=file)  # MAT训练参数
print("Adversarial_Training_type:", "MAT", file=file)
print("Adversarial_init_type:", args["adv_init_type"], file=file)
print("Adversarial_init_epsilon:", args["adv_init_epsilon"], file=file)
print("Warm_up:", args["warm_up"], file=file)
print("Sampling_times_theta:", args["sampling_times_theta"], file=file)
print("Sampling_times_delta:", args["sampling_times_delta"], file=file)
print("Sampling_step_theta:", args["sampling_step_theta"], file=file)
print("Sampling_step_delta:", args["sampling_step_delta"], file=file)
print("Sampling_noise_ratio:", args["sampling_noise_ratio"], file=file)
print("Lambda:", args["lambda_s"], file=file)
print("Beta_s:", args["beta_s"], file=file)
print("Beta_p:", args["beta_p"], file=file)
print("eval_times:", args["eval_times"], file=file)
print("*"*50, file=file)
file.flush()
for epoch in range(args["epochs"]):

    progress_bar.set_description("Training ["+str(epoch+1)+"/"+str(args["epochs"])+"]")
    ###################  Train-begin  ###################
    print("-"*20, "EPOCH:", epoch+1, "-"*20, file=file)
    for batch in train_dataloader:
        model.train()
        batch = {key: batch[key].to(device) for key in batch}

        # [begin] MAT Training
        # 1.init
        # 1.1 备份模型参数
        back_parameters = model.state_dict()  # 备份上一次迭代的模型参数
        mean_theta = model.state_dict()  # 初始化本次迭代的模型参数均值

        # 1.2 初始化模型输入inputs (不包含「inputs_embeds」或「input_ids」，input_ids是词字典ID，inputs_embeds是词向量)
        if "bert-" in args["model_name"]:  # bert模型输入inputs: "attention_mask","labels","token_type_ids", 此外「inputs_embeds」和「input_ids」参数二选一
            inputs = {"attention_mask": batch["attention_mask"], "labels": batch["labels"], "token_type_ids": batch["token_type_ids"]}
            word_embedding = model.bert.embeddings.word_embeddings(batch["input_ids"])  # 获得batch的word_embedding
        # roberta模型输入inputs: "attention_mask","labels", 此外「inputs_embeds」和「input_ids」参数二选一
        elif "roberta-" in args["model_name"]:
            inputs = {"attention_mask": batch["attention_mask"], "labels": batch["labels"]}
            word_embedding = model.roberta.embeddings.word_embeddings(batch["input_ids"])  # 获得batch的word_embedding

        # 1.3 初始化「抽样扰动delta」和「扰动分布均值mean_delta」
        delta = function.init_delta(word_embedding.size(), args["adv_init_epsilon"], args["adv_init_type"])
        mean_delta = delta.clone().detach()  # 初始化delta的分布均值mean_delta

        # 2.stochastic gradient langevin dynamics sampling
        # 2.1 sampling perturbation (delta)
        output_normal_logits = model(**batch).logits.detach()
        for k in range(args["sampling_times_delta"]):
            # 构造带有扰动的inputs_embeds输入
            delta.requires_grad = True
            inputs["inputs_embeds"] = delta + word_embedding.detach()
            # 前向传播
            output_adv = model(**inputs)
            loss_adv = function.ls(output_normal_logits, output_adv.logits, args["task_name"])
            # 反向传播
            loss_adv.backward()
            # SGLD采样
            noise_epsion = args["sampling_step_delta"] * args["sampling_noise_ratio"]
            delta = function.SGLD(delta.detach(), - delta.grad, args["sampling_step_delta"], noise_epsion).detach()
            # 更新扰动的分布均值
            mean_delta = args["beta_s"] * mean_delta + (1 - args["beta_s"]) * delta
        
        # 2.2 sampling model parameters (theta)
        for k in range(args["sampling_times_theta"]):
            # 清空模型参数的梯度
            for p in model.parameters():
                if p.grad != None:
                    p.grad.zero_()
            # 构造带有扰动的输入
            if "bert-" in args["model_name"]:  # 每次backward()会丢失word_embedding的计算图，因此需要每次计算一遍word_embedding
                word_embedding = model.bert.embeddings.word_embeddings(batch["input_ids"])
            elif "roberta-" in args["model_name"]:
                word_embedding = model.roberta.embeddings.word_embeddings(batch["input_ids"])
            inputs["inputs_embeds"] = mean_delta.detach() + word_embedding
            # 前向传播
            output_normal = model(**batch)  # model参数每次都会改变，因此需要重新计算正常输出
            output_normal_logits = output_normal.logits.detach()
            output_adv = model(**inputs)
            loss_adv = function.ls(output_normal_logits, output_adv.logits, args["task_name"])
            loss_sum = output_normal.loss + args["lambda_s"] * loss_adv
            # 反向传播
            loss_sum.backward()
            # SGLD采样并更新分布均值
            for name, p in model.named_parameters():
                sampling_step = function.dynamic_rate(total_iterations, current_iteration, args["sampling_step_theta"], args["warm_up"])
                noise_epsion = args["sampling_step_theta"] * args["sampling_noise_ratio"]
                p.data = function.SGLD(p.data, p.grad, sampling_step, noise_epsion)  # 将模型参数更新为新的采样
                mean_theta[name] = args["beta_s"] * mean_theta[name] + (1 - args["beta_s"]) * p.data  # 更新模型参数的分布均值
        
        # 3.update model parameters
        for key in back_parameters:
            back_parameters[key] = args["beta_p"] * back_parameters[key] + (1 - args["beta_p"]) * mean_theta[key]  # 调整备份的模型参数
        model.load_state_dict(back_parameters)  # 将模型参数更新为这次迭代的模型参数
        # [end] MAT Training
    ###################  Train-end  ###################

    ###################  eval-begin  ###################
        if current_iteration > 0 and current_iteration % eval_step == 0:
            model.eval()
            for batch in eval_dataloader:
                batch = {key: batch[key].to(device) for key in batch}
                with torch.no_grad():
                    outputs = model(**batch)
                predictions = outputs.logits.argmax(dim=-1) if args["task_name"] != "STS-B" else outputs.logits.squeeze()
                metric.add_batch(predictions=predictions, references=batch["labels"])
            metric_data = metric.compute()
            eval_metric_list.append(metric_data)
            score = list(metric_data.values())[0]
            eval_score_list.append(score)
            nni.report_intermediate_result(score)
            print("Iteration:", current_iteration, "Metric:", metric_data, file=file)
    ###################  eval-end  ###################

    ###################  Test-begin  ###################
            if args["predict"] == "True" and (args["task_name"] == "ANLI" or score == max(eval_score_list)):
                with(open(log_path + "/" + args["task_name"]+".tsv", "w")) as f_tsv:
                    tsv_writer = csv.writer(f_tsv, delimiter='\t')
                    tsv_writer.writerow(["index", "prediction"])
                    for t, batch in enumerate(test_dataloader):
                        batch = {key: batch[key].to(device) for key in batch}
                        if args["task_name"]!="ANLI":
                            batch['labels'] = None # model的outputs默认会利用batch中的labels计算loss，测试集label默认为-1，会导致报错，labels设为None可规避该问题
                        with torch.no_grad():
                            outputs = model(**batch)
                        predictions = outputs.logits.argmax(dim=-1) if args["task_name"] != "STS-B" else outputs.logits.squeeze()
                        if args["task_name"]=="ANLI":
                            metric.add_batch(predictions=predictions, references=batch["labels"])
                        for id, pre in enumerate(predictions):
                            predict_label = preprocess.TASKS_TO_LABELS[args["task_name"]][pre.item()] if args["task_name"] != "STS-B" else round(pre.item(), 3)  # 保留3位小数
                            tsv_writer.writerow([id + t * args["batch_size"], predict_label])
                    if args["task_name"]=="ANLI":
                        metric_data = metric.compute()
                        print("Iteration:", current_iteration, "Test Metric:", metric_data, file=file)
                    f_tsv.close()
            print("-"*50, file=file)
    ###################  Test-end  ###################
            file.flush()
        current_iteration = current_iteration + 1
        progress_bar.update(1)

# Best score in validation set
nni.report_final_result(max(eval_score_list))
print("*"*19, "Best Score", "*"*19, file=file)
print("Best Metric:", eval_metric_list[eval_score_list.index(max(eval_score_list))], file=file)
print("*"*50, file=file)
file.close()

csv_file = open("logs/" + args["task_name"] + "/" + args["task_name"] + "_" + args["model_name"] + ".csv", "a") 
print(run_time + "," + str(args["seed"]) + "," + str(args["epochs"]) + "," + str(args["batch_size"]) + "," + str(args["adv_init_type"]) + "," + str(args["adv_init_epsilon"]) + "," + str(args["warm_up"]) + "," + str(args["sampling_times_theta"]) + "," + str(args["sampling_times_delta"]) + "," + str(args["sampling_step_theta"]) + "," + str(args["sampling_step_delta"]) + "," + str(args["sampling_noise_ratio"]) + "," + str(args["lambda_s"]) + "," + str(args["beta_s"]) + "," + str(args["beta_p"]) + "," + str(args["eval_times"]) + "," + str(max(eval_score_list)), file=csv_file)
csv_file.close()