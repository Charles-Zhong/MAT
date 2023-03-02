PY_FILE=MAT.py

TASK_NAME=CoLA
MODEL_NAME=bert-base-uncased

SEED=42
BATCH_SIZE=64
EPOCHS=10

OPTIMIZER=Adam

SAMPLING_TIMES_THETA=10
SAMPLING_TIMES_DELTA=10

SAMPLING_STEP_THETA=1e-5
SAMPLING_STEP_DELTA=1e-3
SAMPLING_NOISE_RATIO=0.2

LAMBDA_S=3
BETA_S=0.3
BETA_P=0.3

EVAL_TIMES=5

python ${PY_FILE} --task_name ${TASK_NAME} --model_name ${MODEL_NAME} \
                  --seed ${SEED} --batch_size ${BATCH_SIZE} --epochs ${EPOCHS} --optimizer ${OPTIMIZER}\
                  --sampling_times_theta ${SAMPLING_TIMES_THETA} --sampling_times_delta ${SAMPLING_TIMES_DELTA} \
                  --sampling_step_theta ${SAMPLING_STEP_THETA} --sampling_step_delta ${SAMPLING_STEP_DELTA} \
                  --sampling_noise_ratio ${SAMPLING_NOISE_RATIO} \
                  --lambda_s ${LAMBDA_S} --beta_s ${BETA_S} --beta_p ${BETA_P} --eval_times ${EVAL_TIMES}