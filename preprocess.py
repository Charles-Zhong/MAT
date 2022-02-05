from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, set_seed

# 设置任务数据集的映射关系
GLUE_TASKS = ["CoLA", "SST-2", "MRPC", "STS-B", "QQP",
              "MNLI-m", "MNLI-mm", "QNLI", "RTE", "WNLI"]
HUGGINGFACE_GLUE_TASKS = ["cola", "sst2", "mrpc", "stsb",
                          "qqp", "mnli", "mnli", "qnli", "rte", "wnli"]
TASKS_TO_KEYS = {
    "CoLA": ("sentence", None),
    "SST-2": ("sentence", None),
    "MRPC": ("sentence1", "sentence2"),
    "STS-B": ("sentence1", "sentence2"),
    "QQP": ("question1", "question2"),
    "MNLI-m": ("premise", "hypothesis"),
    "MNLI-mm": ("premise", "hypothesis"),
    "QNLI": ("question", "sentence"),
    "RTE": ("sentence1", "sentence2"),
    "WNLI": ("sentence1", "sentence2")
}


def LoadDataset(task_name):
    # 加载数据集
    print("-" * 18, "load dataset", "-" * 18)
    print("[Notice]: loading dataset...")
    actual_task = HUGGINGFACE_GLUE_TASKS[GLUE_TASKS.index(task_name)]
    raw_dataset = load_dataset("datasets/glue", actual_task)
    metric = load_metric("metrics/glue", actual_task)
    print("[Notice]: dataset", actual_task, "loaded.")
    print("-" * 50)
    return raw_dataset, metric


def LoadModel(task_name, model_name):
    # 加载tokenizer和model
    print("-" * 12, "load tokenizer and model", "-" * 12)
    print("[Notice]: loading tokenizer and model...")
    # 根据task设置类别数
    if task_name == "MNLI-m" or task_name == "MNLI-mm":
        num_labels = 3  # 3分类
    elif task_name == "STS-B":
        num_labels = 1  # 回归
    else:
        num_labels = 2  # 2分类
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    print("[Notice]: tokenizer and model loaded.")
    print("-" * 50)
    return tokenizer, model


def Tokenize(task_name, tokenizer, raw_dataset):
    print("-" * 13, "tokenize the dataset", "-" * 13)
    print("[Notice]: tokenizing the dataset...")
    sentence1_key, sentence2_key = TASKS_TO_KEYS[task_name]
    # Tokenize文本
    def preprocess_function(examples):
        texts = ((examples[sentence1_key],) if sentence2_key is None else (
            examples[sentence1_key], examples[sentence2_key]))
        result = tokenizer(*texts)
        result["labels"] = examples["label"]
        return result
    tokenized_dataset = raw_dataset.map(preprocess_function, batched=True, remove_columns=raw_dataset["train"].column_names)
    print("[Notice]: the dataset is tokenized.")
    print("-" * 50)
    return tokenized_dataset


def MakeDataloader(tokenized_dataset, task_name, tokenizer, BATCH_SIZE):
    # 设置训练集、验证集、测试集
    print("-" * 17, "make dataloader", "-" * 18)
    print("[Notice]: making dataloader...")
    train_dataset = tokenized_dataset["train"]
    if task_name == "MNLI-m":
        val_dataset = tokenized_dataset["validation_matched"]
        test_dataset = tokenized_dataset["test_matched"]
    elif task_name == "MNLI-mm":
        val_dataset = tokenized_dataset["validation_mismatched"]
        test_dataset = tokenized_dataset["test_mismatched"]
    else:
        val_dataset = tokenized_dataset["validation"]
        test_dataset = tokenized_dataset["test"]
    # 创建dataloader
    data_collator = DataCollatorWithPadding(tokenizer)
    train_dataloader = DataLoader(train_dataset, collate_fn=data_collator, shuffle=True, batch_size=BATCH_SIZE)
    eval_dataloader = DataLoader(val_dataset, collate_fn=data_collator, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=BATCH_SIZE)
    print("[Notice]: make dataloader is done.")
    print("-" * 50)
    return (train_dataloader, eval_dataloader, test_dataloader)


def preprocess(task_name, model_name, BATCH_SIZE, SEED):
    set_seed(SEED)
    raw_dataset, metric = LoadDataset(task_name)
    tokenizer, model = LoadModel(task_name, model_name)
    tokenized_dataset = Tokenize(task_name, tokenizer, raw_dataset)
    dataloader = MakeDataloader(tokenized_dataset, task_name, tokenizer, BATCH_SIZE)
    return model, dataloader, metric
