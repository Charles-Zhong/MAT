from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

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
TASKS_TO_LABEL = {
    "CoLA": ("0", "1"),
    "SST-2": ("0", "1"),
    "MRPC": ("0", "1"),
    "QQP": ("0", "1"),
    "MNLI-m": ("entailment", "neutral", "contradiction"),
    "MNLI-mm": ("entailment", "neutral", "contradiction"),
    "QNLI": ("entailment", "not_entailment"),
    "RTE": ("entailment", "not_entailment"),
    "WNLI": ("0", "1")
}

def LoadDataset(task_name):
    # 加载数据集
    print("-" * 16, "load the dataset", "-" * 16)
    print("[Notice]: loading dataset...")
    actual_task = HUGGINGFACE_GLUE_TASKS[GLUE_TASKS.index(task_name)]
    raw_dataset = load_dataset("datasets/glue", actual_task)
    metric = load_metric("metrics/glue", actual_task)
    num_labels = len(raw_dataset["train"].features["label"].names) if task_name!='STS-B' else 1 # 根据task设置类别数
    print("[Notice]: dataset", actual_task, "is loaded.")
    print("-" * 50)
    return raw_dataset, metric, num_labels

def LoadModel(task_name, model_name, num_labels):
    # 加载tokenizer和model
    print("-" * 8, "load the tokenizer and the model", "-" * 8)
    print("[Notice]: loading tokenizer and model...")
    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    print("[Notice]: tokenizer and model are loaded.")
    print("-" * 50)
    return tokenizer, model

def Tokenize(task_name, tokenizer, raw_dataset):
    print("-" * 14, "tokenize the dataset", "-" * 14)
    print("[Notice]: tokenizing the dataset...")
    sentence1_key, sentence2_key = TASKS_TO_KEYS[task_name]
    # Tokenize文本
    def preprocess_function(examples):
        texts = ((examples[sentence1_key],) if sentence2_key is None else (
            examples[sentence1_key], examples[sentence2_key]))
        result = tokenizer(*texts)
        result["labels"] = examples["label"]
        return result
    tokenized_dataset = raw_dataset.map(preprocess_function, remove_columns=raw_dataset["train"].column_names, batched=True, keep_in_memory=True)
    print("[Notice]: the dataset is tokenized.")
    print("-" * 50)
    return tokenized_dataset

def MakeDataloader(tokenized_dataset, task_name, tokenizer, batch_size):
    # 设置训练集、验证集、测试集
    print("-" * 14, "make the dataloader", "-" * 15)
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
    train_dataloader = DataLoader(train_dataset, collate_fn=data_collator, shuffle=True, batch_size=batch_size)
    eval_dataloader = DataLoader(val_dataset, collate_fn=data_collator, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=batch_size)
    print("[Notice]: the dataloader is made.")
    print("-" * 50)
    return (train_dataloader, eval_dataloader, test_dataloader)

def preprocess(task_name, model_name, batch_size):
    raw_dataset, metric, num_labels = LoadDataset(task_name)
    tokenizer, model = LoadModel(task_name, model_name, num_labels)
    tokenized_dataset = Tokenize(task_name, tokenizer, raw_dataset)
    dataloader = MakeDataloader(tokenized_dataset, task_name, tokenizer, batch_size)
    return model, dataloader, metric