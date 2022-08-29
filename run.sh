# usage: XXX-MAT.py [-h] [--task_name {CoLA,SST-2,MRPC,STS-B,QQP,MNLI-m,MNLI-mm,QNLI,RTE,WNLI,CQA,ANLI}]]
#                    [--model_path MODEL_PATH] [--model_name {bert-base-uncased,roberta-large}]
#                    [--seed SEED] [--batch_size BATCH_SIZE] [--epochs EPOCHS]
#                    [--adv_init_type {zero,rand,randn}] [--adv_init_epsilon ADV_INIT_EPSILON]
#                    [--sampling_times_theta SAMPLING_TIMES_THETA] [--sampling_times_delta SAMPLING_TIMES_DELTA]
#                    [--sampling_step_theta SAMPLING_STEP_THETA] [--sampling_step_delta SAMPLING_STEP_DELTA]
#                    [--sampling_noise_ratio SAMPLING_NOISE_RATIO] [--lambda_s LAMBDA_S]
#                    [--beta_s BETA_S] [--beta_p BETA_P] [--save_model SAVE_MODEL]
#!/bin/bash

PY_FILE=MAT.py

TASK_NAME=MRPC
MODEL_PATH=models
MODEL_NAME=roberta-large

SEED=8888
BATCH_SIZE=12
EPOCHS=10

ADV_INIT_TYPE=randn
ADV_INIT_EPSILON=5e-5

SAMPLING_TIMES_THETA=10
SAMPLING_TIMES_DELTA=10

SAMPLING_STEP_THETA=5e-4
SAMPLING_STEP_DELTA=5e-4
SAMPLING_NOISE_RATIO=0.01

LAMBDA_S=1
BETA_S=0.5
BETA_P=0.5

python ${PY_FILE} --task_name ${TASK_NAME} --model_path ${MODEL_PATH} --model_name ${MODEL_NAME} \
                  --seed ${SEED} --batch_size ${BATCH_SIZE} --epochs ${EPOCHS} \
                  --adv_init_type ${ADV_INIT_TYPE} --adv_init_epsilon ${ADV_INIT_EPSILON} \
                  --sampling_times_theta ${SAMPLING_TIMES_THETA} --sampling_times_delta ${SAMPLING_TIMES_DELTA} \
                  --sampling_step_theta ${SAMPLING_STEP_THETA} --sampling_step_delta ${SAMPLING_STEP_DELTA} \
                  --sampling_noise_ratio ${SAMPLING_NOISE_RATIO}\
                  --lambda_s ${LAMBDA_S} --beta_s ${BETA_S} --beta_p ${BETA_P} 