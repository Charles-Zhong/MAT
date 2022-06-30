# usage: GLUE-MAT.py [-h] [--task_name {CoLA,SST-2,MRPC,STS-B,QQP,MNLI-m,MNLI-mm,QNLI,RTE,WNLI}]
#                    [--model_name {bert-base-uncased,roberta-base,roberta-large}] [--seed SEED]
#                    [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--adv_init_epsilon ADV_INIT_EPSILON] [--adv_init_type ADV_INIT_TYPE]
#                    [--sampling_times_theta SAMPLING_TIMES_THETA] [--sampling_times_delta SAMPLING_TIMES_DELTA]
#                    [--sampling_noise_theta SAMPLING_NOISE_THETA] [--sampling_noise_delta SAMPLING_NOISE_DELTA]
#                    [--sampling_step_theta SAMPLING_STEP_THETA] [--sampling_step_delta SAMPLING_STEP_DELTA]
#                    [--lambda_s LAMBDA_S] [--beta_s BETA_S] [--beta_p BETA_P] [--save_model SAVE_MODEL]

python GLUE-MAT.py