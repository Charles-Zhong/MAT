# usage: GLUE-MAT.py [-h] [--task_name {CoLA,SST-2,MRPC,STS-B,QQP,MNLI-m,MNLI-mm,QNLI,RTE,WNLI}]
#                    [--model_name {bert-base-uncased,roberta-base,roberta-large}]
#                    [--seed SEED] [--batch_size BATCH_SIZE] [--epochs EPOCHS]
#                    [--adv_init_epsilon ADV_INIT_EPSILON] [--adv_init_type {zero,rand,randn}]
#                    [--sampling_times_theta SAMPLING_TIMES_THETA] [--sampling_times_delta SAMPLING_TIMES_DELTA]
#                    [--sampling_step_theta SAMPLING_STEP_THETA] [--sampling_step_delta SAMPLING_STEP_DELTA]
#                    [--sampling_noise_theta SAMPLING_NOISE_THETA] [--sampling_noise_delta SAMPLING_NOISE_DELTA]
#                    [--lambda_s LAMBDA_S] [--beta_s BETA_S] [--beta_p BETA_P] [--save_model SAVE_MODEL]
#!/bin/bash
PY_FILE=GLUE_MAT_EVAL+.py

TASK_NAME=STS-B
MODEL_NAME=bert-base-uncased

SEED=42
BATCH_SIZE=32
EPOCHS=10

ADV_INIT_TYPE=zero
ADV_INIT_EPSILON=0.0013

SAMPLING_TIMES_THETA=20
SAMPLING_TIMES_DELTA=5

SAMPLING_STEP_THETA=0.001
SAMPLING_STEP_DELTA=0.0015

SAMPLING_NOISE_THETA=0.00005
SAMPLING_NOISE_DELTA=0.00006

LAMBDA_S=0.1
BETA_S=0.4
BETA_P=0.3

python ${PY_FILE} --task_name ${TASK_NAME} --model_name ${MODEL_NAME} \
                  --seed ${SEED} --batch_size ${BATCH_SIZE} --epochs ${EPOCHS} \
                  --adv_init_type ${ADV_INIT_TYPE} --adv_init_epsilon ${ADV_INIT_EPSILON} \
                  --sampling_times_theta ${SAMPLING_TIMES_THETA} --sampling_times_delta ${SAMPLING_TIMES_DELTA} \
                  --sampling_step_theta ${SAMPLING_STEP_THETA} --sampling_step_delta ${SAMPLING_STEP_DELTA} \
                  --sampling_noise_theta ${SAMPLING_NOISE_THETA} --sampling_noise_delta ${SAMPLING_NOISE_DELTA} \
                  --lambda_s ${LAMBDA_S} --beta_s ${BETA_S} --beta_p ${BETA_P} 