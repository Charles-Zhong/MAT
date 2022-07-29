import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a model with MAT training mode.")
    parser.add_argument("--task_name", type=str, default="CoLA", 
                        choices=["CoLA", "SST-2", "MRPC", "STS-B", "QQP", "MNLI-m", "MNLI-mm", "QNLI", "RTE", "WNLI", "CQA", "ANLI"], 
                        help="The name of the task.")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", 
                        choices=["bert-base-uncased", "roberta-large", "roberta-large-mnli"], 
                        help="Finetune base model.")
    parser.add_argument("--seed", type=int, default=42,
                        help="A seed for reproducible training.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for the dataloader.")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Total number of training epochs.")
    parser.add_argument("--adv_init_type", type=str, default="randn", 
                        choices=["zero", "rand", "randn"],
                        help="Initialization type of adversarial perturbation.")
    parser.add_argument("--adv_init_epsilon", type=float, default=0.001,
                        help="Initialization size of adversarial perturbation.")
    parser.add_argument("--sampling_times_theta", type=int, default=30,
                        help="SGLD sampling times for model parameters.")
    parser.add_argument("--sampling_times_delta", type=int, default=5,
                        help="SGLD sampling times for adversarial perturbation.")
    parser.add_argument("--sampling_step_theta", type=float, default=0.001,
                        help="SGLD sampling step for model parameters.")
    parser.add_argument("--sampling_step_delta", type=float, default=0.001,
                        help="SGLD sampling step for adversarial perturbation.")
    parser.add_argument("--sampling_noise_theta", type=float, default=1e-5,
                        help="SGLD sampling noise for model parameters.")
    parser.add_argument("--sampling_noise_delta", type=float, default=1e-5,
                        help="SGLD sampling noise for adversarial perturbation.")
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
