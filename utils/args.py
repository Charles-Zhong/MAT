import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a model with MAT training mode.")
    parser.add_argument("--task_name", type=str, default="CoLA", 
                        choices=["CoLA", "SST-2", "MRPC", "STS-B", "QQP", "MNLI-m", "MNLI-mm", "QNLI", "RTE", "WNLI", "CQA", "ANLI"], 
                        help="The name of the task.")
    parser.add_argument("--model_path", type=str, default="",
                        help="Local model path. If it is empty, it will be downloaded automatically.")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", 
                        choices=["bert-base-uncased", "roberta-large"], 
                        help="Finetune base model.")
    parser.add_argument("--seed", type=int, default=0,
                        help="A seed for recurrence experiment.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for the dataloader.")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Total number of training epochs.")
    parser.add_argument("--adv_init_type", type=str, default="randn", 
                        choices=["zero", "rand", "randn"],
                        help="Init type of adversarial perturbation.")
    parser.add_argument("--adv_init_epsilon", type=float, default=1e-5,
                        help="Init size of adversarial perturbation.")
    parser.add_argument("--sampling_times_theta", type=int, default=10,
                        help="SGLD sampling times for model parameters.")
    parser.add_argument("--sampling_times_delta", type=int, default=10,
                        help="SGLD sampling times for adversarial perturbation.")
    parser.add_argument("--sampling_step_theta", type=float, default=1e-5,
                        help="SGLD sampling step for model parameters.")
    parser.add_argument("--sampling_step_delta", type=float, default=1e-5,
                        help="SGLD sampling step for adversarial perturbation.")
    parser.add_argument("--sampling_noise_ratio", type=float, default=0.01,
                        help="SGLD sampling noise ratio.")
    parser.add_argument("--lambda_s", type=float, default=1,
                        help="Tuning parameter lambda of the objective function.")
    parser.add_argument("--beta_s", type=float, default=0.5,
                        help="Exponential damping beta for stability in SGLD sampling.")
    parser.add_argument("--beta_p", type=float, default=0.5,
                        help="Exponential damping beta for stability in parameters updating.")
    parser.add_argument("--save_model", action="store_true",
                        help="Save the best model during training.")
    args = parser.parse_args()
    return args
