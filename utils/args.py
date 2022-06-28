import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on GLUE with MAT training mode.")
    parser.add_argument("--task_name", type=str, default="SST-2", 
                        choices=["CoLA", "SST-2", "MRPC", "STS-B", "QQP", "MNLI-m", "MNLI-mm", "QNLI", "RTE", "WNLI"], 
                        help="The name of the glue task to train on.")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", 
                        choices=["bert-base-uncased", "roberta-base", "roberta-large"], 
                        help="Finetune base model.")
    parser.add_argument("--seed", type=int, default=42,
                        help="A seed for reproducible training.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for the training dataloader.")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--adv_init_epsilon", type=float, default=0.001,
                        help="Initialization size of adversarial perturbation.")
    parser.add_argument("--adv_init_type", type=str, default="randn", 
                        choices=["zero", "rand", "randn"],
                        help="Initialization type of adversarial perturbation.")
    parser.add_argument("--sampling_times_theta", type=int, default=30,
                        help="Stochastic gradient langevin dynamics sampling times for model parameters.")
    parser.add_argument("--sampling_times_delta", type=int, default=5,
                        help="Stochastic gradient langevin dynamics sampling times for adversarial perturbation.")
    parser.add_argument("--sampling_noise_theta", type=float, default=1e-5,
                        help="Stochastic gradient langevin dynamics sampling noise for model parameters.")
    parser.add_argument("--sampling_noise_delta", type=float, default=1e-5,
                        help="Stochastic gradient langevin dynamics sampling noise for adversarial perturbation.")
    parser.add_argument("--sampling_step_theta", type=float, default=0.001,
                        help="Stochastic gradient langevin dynamics sampling step for model parameters.")
    parser.add_argument("--sampling_step_delta", type=float, default=0.001,
                        help="Stochastic gradient langevin dynamics sampling step for adversarial perturbation.")
    parser.add_argument("--lambda_s", type=float, default=1,
                        help="Tuning parameter lambda of the objective function.")
    parser.add_argument("--beta_s", type=float, default=0.5,
                        help="Exponential damping beta for stability in stochastic gradient langevin dynamics sampling.")
    parser.add_argument("--beta_p", type=float, default=0.5,
                        help="Exponential damping beta for stability in parameters updating.")
    parser.add_argument("--save_model", type=bool, default=False,
                        help="Save the best model during training.")
    args = parser.parse_args()
    return args
