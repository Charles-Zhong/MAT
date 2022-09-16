# usage: MAT.py      [-h] [--task_name {CoLA,SST-2,MRPC,STS-B,QQP,MNLI-m,MNLI-mm,QNLI,RTE,WNLI,ANLI}]]
#                    [--model_path MODEL_PATH] [--model_name] [--seed SEED] [--batch_size BATCH_SIZE] [--epochs EPOCHS]
#                    [--adv_init_type {zero,rand,randn}] [--adv_init_epsilon ADV_INIT_EPSILON] [--warm_up WARM_UP]
#                    [--sampling_times_theta SAMPLING_TIMES_THETA] [--sampling_times_delta SAMPLING_TIMES_DELTA]
#                    [--sampling_step_theta SAMPLING_STEP_THETA] [--sampling_step_delta SAMPLING_STEP_DELTA]
#                    [--sampling_noise_ratio SAMPLING_NOISE_RATIO] [--lambda_s LAMBDA_S]
#                    [--beta_s BETA_S] [--beta_p BETA_P] [--eval_times EVAL_TIMES] [--predict PREDICT]
#!/bin/bash

PY_FILE=MAT.py

TASK_NAME=CoLA
MODEL_PATH=models
MODEL_NAME=roberta-large

SEED=8888
BATCH_SIZE=32
EPOCHS=10

ADV_INIT_TYPE=randn
ADV_INIT_EPSILON=0.0002

WARM_UP=0.1

SAMPLING_TIMES_THETA=20
SAMPLING_TIMES_DELTA=10

SAMPLING_STEP_THETA=0.0015
SAMPLING_STEP_DELTA=0.0015
SAMPLING_NOISE_RATIO=0.01

LAMBDA_S=1
BETA_S=0.6
BETA_P=0.4

EVAL_TIMES=10

python ${PY_FILE} --task_name ${TASK_NAME} --model_path ${MODEL_PATH} --model_name ${MODEL_NAME} \
                  --seed ${SEED} --batch_size ${BATCH_SIZE} --epochs ${EPOCHS} \
                  --adv_init_type ${ADV_INIT_TYPE} --adv_init_epsilon ${ADV_INIT_EPSILON} --warm_up ${WARM_UP} \
                  --sampling_times_theta ${SAMPLING_TIMES_THETA} --sampling_times_delta ${SAMPLING_TIMES_DELTA} \
                  --sampling_step_theta ${SAMPLING_STEP_THETA} --sampling_step_delta ${SAMPLING_STEP_DELTA} \
                  --sampling_noise_ratio ${SAMPLING_NOISE_RATIO} \
                  --lambda_s ${LAMBDA_S} --beta_s ${BETA_S} --beta_p ${BETA_P} --eval_times ${EVAL_TIMES}