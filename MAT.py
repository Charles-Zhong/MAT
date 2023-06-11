import os
import csv
import time
import torch
import random
import transformers
import numpy as np
from tqdm.auto import tqdm
import utils

args = utils.argparse.parse_args()

# set seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

run_time = str(int(time.time()))
log_path = "logs/" + args.task_name + "/" + args.model_name + "/" + run_time
os.makedirs(log_path)
file = open(log_path + "/" + args.task_name + "_" + args.model_name + "_" + run_time + ".log", "w")

local_model_path = args.model_path  + "/" if args.model_path != "" else ""
model, dataloader, metric = utils.process(args.task_name, local_model_path + args.model_name, args.batch_size)
train_dataloader, eval_dataloader, test_dataloader = dataloader
model.to(device)

current_iteration = 0
total_iterations = args.epochs * len(train_dataloader)

if args.optimizer == "SGD":
    optim = torch.optim.SGD
elif args.optimizer == "RMSprop":
    optim = torch.optim.RMSprop
elif args.optimizer == "Adam":
    optim = torch.optim.AdamW
optimizer = optim(model.parameters(), lr=args.sampling_step_theta)
lr_scheduler = transformers.get_scheduler(name=args.scheduler_type, optimizer=optimizer, num_warmup_steps=args.warm_up * total_iterations, num_training_steps=total_iterations)

theta_noise_epsion = args.sampling_step_theta * args.sampling_noise_ratio
delta_noise_epsion = args.sampling_step_delta * args.sampling_noise_ratio

progress_bar = tqdm(range(total_iterations))
eval_step = len(train_dataloader) // args.eval_times
eval_metric_list = []
eval_score_list = []

# Training
print(time.ctime(), file=file)
print("*"*20, "Training", "*"*20, file=file)
print("TASK:", args.task_name, file=file)
print("MODEL:", args.model_name, file=file)
print("="*16, "General Training", "="*16, file=file)
print("SEED:", args.seed, file=file)
print("EPOCHS:", args.epochs, file=file)
print("BATCH_SIZE:", args.batch_size, file=file)
print("OPTIMIZER:", args.optimizer, file=file)
print("LR_SCHEDULER:", args.scheduler_type, file=file)
print("WARM_UP_RATIO:", args.warm_up, file=file)
print("="*18, "MAT Training", "="*18, file=file)
print("ADV_INIT_TYPE:", args.adv_init_type, file=file)
print("ADV_INIT_EPSILON:", args.adv_init_epsilon, file=file)
print("ADV_MAX_NORM:", args.adv_max_norm, file=file)
print("SAMPLING_TIMES_THETA:", args.sampling_times_theta, file=file)
print("SAMPLING_TIMES_DELTA:", args.sampling_times_delta, file=file)
print("SAMPLING_STEP_THETA:", args.sampling_step_theta, file=file)
print("SAMPLING_STEP_DELTA:", args.sampling_step_delta, file=file)
print("SAMPLING_NOISE_RATIO:", args.sampling_noise_ratio, file=file)
print("LAMBDA:", args.lambda_s, file=file)
print("BETA_S:", args.beta_s, file=file)
print("BETA_P:", args.beta_p, file=file)
print("EVAL_TIMES:", args.eval_times, file=file)
print("*"*50, file=file)
file.flush()

for epoch in range(args.epochs):

    progress_bar.set_description("Training ["+str(epoch+1)+"/"+str(args.epochs)+"]")
    ###################  Train-begin  ###################
    print("-"*20, "EPOCH:", epoch+1, "-"*20, file=file)
    for batch in train_dataloader:
        model.train()
        batch = {key: batch[key].to(device) for key in batch}

        # [begin] MAT Training
        if "bert-" in args.model_name:
            inputs = {"attention_mask": batch["attention_mask"], "labels": batch["labels"], "token_type_ids": batch["token_type_ids"]}
            word_embedding = model.bert.embeddings.word_embeddings(batch["input_ids"])
        elif "roberta-" in args.model_name:
            inputs = {"attention_mask": batch["attention_mask"], "labels": batch["labels"]}
            word_embedding = model.roberta.embeddings.word_embeddings(batch["input_ids"])
        word_embedding.detach_()

        # 1. update distribution of perturbation (delta)
        delta = utils.init_delta(word_embedding.size(), args.adv_init_epsilon, args.adv_init_type)
        mean_delta = delta.clone().detach()

        output_normal_logits = model(**batch).logits.detach()
        for k in range(args.sampling_times_delta):
            delta.requires_grad = True
            inputs["inputs_embeds"] = delta + word_embedding # x+δ_t^k
            output_adv = model(**inputs) # f(x+δ_t^k; θ_t)
            loss_adv = utils.ls(output_normal_logits, output_adv.logits, args.task_name) # ls(f(x+δ_t^k; θ_t),f(x; θ_t))
            loss_adv.backward()
            delta = utils.update_delta(delta, args.sampling_step_delta, args.adv_max_norm).detach()
            delta = utils.SGLD(delta, args.sampling_step_delta, delta_noise_epsion).detach()
            mean_delta = args.beta_s * mean_delta + (1 - args.beta_s) * delta
        mean_delta.detach_()

        # 2. update distribution of model parameters (theta)
        back_parameters = model.state_dict()
        mean_theta = model.state_dict()
        for k in range(args.sampling_times_theta):
            optimizer.zero_grad()
            if "bert-" in args.model_name:
                word_embedding = model.bert.embeddings.word_embeddings(batch["input_ids"])
            elif "roberta-" in args.model_name:
                word_embedding = model.roberta.embeddings.word_embeddings(batch["input_ids"])
            inputs["inputs_embeds"] = mean_delta + word_embedding
            output_normal = model(**batch)  # f(x; θ_t^k)
            output_adv = model(**inputs) # f(x+δ_t; θ_t^k)
            loss_adv = utils.ls(output_normal.logits, output_adv.logits, args.task_name) # ls(f(x; θ_t^k), f(x+δ_t; θ_t^k))
            loss_sum = output_normal.loss + args.lambda_s * loss_adv # loss1 + λ * loss2
            loss_sum.backward()
            optimizer.step()
            for name, p in model.named_parameters():
                lr_now = lr_scheduler.get_last_lr()[0]
                p.data = utils.SGLD(p.data, lr_now, theta_noise_epsion)
                mean_theta[name] = args.beta_s * mean_theta[name] + (1 - args.beta_s) * p.data
        lr_scheduler.step()
        
        # 3.update model parameters
        for name in back_parameters:
            back_parameters[name] = args.beta_p * back_parameters[name] + (1 - args.beta_p) * mean_theta[name]
        model.load_state_dict(back_parameters)
        # [end] MAT Training
    ###################  Train-end  ###################

    ###################  eval-begin  ###################
        if current_iteration > 0 and current_iteration % eval_step == 0:
            model.eval()
            for batch in eval_dataloader:
                batch = {key: batch[key].to(device) for key in batch}
                with torch.no_grad():
                    outputs = model(**batch)
                predictions = outputs.logits.argmax(dim=-1) if args.task_name != "STS-B" else outputs.logits.squeeze()
                metric.add_batch(predictions=predictions, references=batch["labels"])
            metric_data = metric.compute()
            eval_metric_list.append(metric_data)
            score = list(metric_data.values())[0]
            eval_score_list.append(score)
            print("Iteration:", current_iteration, "Metric:", metric_data, file=file)
    ###################  eval-end  ###################

    ###################  Test-begin  ###################
            if args.predict and (args.task_name == "ANLI" or score == max(eval_score_list)):
                with(open(log_path + "/" + args.task_name+".tsv", "w")) as f_tsv:
                    tsv_writer = csv.writer(f_tsv, delimiter='\t')
                    tsv_writer.writerow(["index", "prediction"])
                    for t, batch in enumerate(test_dataloader):
                        batch = {key: batch[key].to(device) for key in batch}
                        if args.task_name!="ANLI":
                            batch['labels'] = None
                        with torch.no_grad():
                            outputs = model(**batch)
                        predictions = outputs.logits.argmax(dim=-1) if args.task_name != "STS-B" else outputs.logits.squeeze()
                        if args.task_name == "ANLI":
                            metric.add_batch(predictions=predictions, references=batch["labels"])
                        for id, pre in enumerate(predictions):
                            predict_label = utils.preprocess.TASKS_TO_LABELS[args.task_name][pre.item()] if args.task_name != "STS-B" else round(pre.item(), 3)
                            tsv_writer.writerow([id + t * args.batch_size, predict_label])
                    if args.task_name == "ANLI":
                        metric_data = metric.compute()
                        print("Iteration:", current_iteration, "Test Metric:", metric_data, file=file)
                    f_tsv.close()
            print("-"*50, file=file)
    ###################  Test-end  ###################
            file.flush()
        current_iteration += 1
        progress_bar.update(1)

# Best score in validation set
print("*"*19, "Best Score", "*"*19, file=file)
print("Best Metric:", eval_metric_list[eval_score_list.index(max(eval_score_list))], file=file)
print("*"*50, file=file)
file.close()

if not os.path.exists("logs/" + args.task_name + "/" + args.task_name + "_" + args.model_name + ".csv"):
    csv_file = open("logs/" + args.task_name + "/" + args.task_name + "_" + args.model_name + ".csv", "a")
    print("run_time,seed,epochs,batch_size,optimizer,scheduler,warm_up,adv_init,adv_init_epsilon,adv_max_norm,times_theta,times_delta,step_theta,step_delta,noise_ratio,lambda,beta_s,beta_p,eval_times,score", file=csv_file)
csv_file = open("logs/" + args.task_name + "/" + args.task_name + "_" + args.model_name + ".csv", "a") 
print(run_time + "," + str(args.seed) + "," + str(args.epochs) + "," + str(args.batch_size) + "," + str(args.optimizer) + "," + str(args.scheduler_type) + "," + str(args.warm_up) + "," + str(args.adv_init_type) + "," + str(args.adv_init_epsilon) + "," + str(args.adv_max_norm) + "," + str(args.sampling_times_theta) + "," + str(args.sampling_times_delta) + "," + str(args.sampling_step_theta) + "," + str(args.sampling_step_delta) + "," + str(args.sampling_noise_ratio) + "," + str(args.lambda_s) + "," + str(args.beta_s) + "," + str(args.beta_p) + "," + str(args.eval_times) + "," + str(max(eval_score_list)), file=csv_file)
csv_file.close()