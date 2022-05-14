# GLUE-MAT.py [-h] [--task_name {CoLA,SST-2,MRPC,STS-B,QQP,MNLI-m,MNLI-mm,QNLI,RTE,WNLI}]
#                  [--model_name {models/bert-base-uncased,models/roberta-base}] [--seed SEED] [--batch_size BATCH_SIZE] [--epochs EPOCH]
#                  [--adv_init_epsilon ADV_INIT_EPSILON] [--adv_init_type ADV_INIT_TYPE] [--sampling_times_theta SAMPLING_TIMES_THETA]
#                  [--sampling_times_delta SAMPLING_TIMES_DELTA] [--sampling_noise_theta SAMPLING_NOISE_THETA]
#                  [--sampling_noise_delta SAMPLING_NOISE_DELTA] [--sampling_step_theta SAMPLING_STEP_THETA]
#                  [--sampling_step_delta SAMPLING_STEP_DELTA] [--lambda_s LAMBDA_S] [--beta BETA]

python -c "print('■'*100)"
python GLUE-MAT.py --lambda_s 0 --beta 0.1 --sampling_times_theta 10 --sampling_times_delta 0 --epoch 5
python -c "print('■'*100)"
python GLUE-MAT.py --lambda_s 0 --beta 0.2 --sampling_times_theta 10 --sampling_times_delta 0 --epoch 5
python -c "print('■'*100)"