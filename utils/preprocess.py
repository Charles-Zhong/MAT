from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

anli_part = "_r1"
token_max_length = 512 

TASKS_TO_DATASETS = {"CoLA": "glue", "SST-2": "glue", "MRPC": "glue", "STS-B": "glue", "QQP": "glue",
                     "MNLI-m": "glue", "MNLI-mm": "glue", "QNLI": "glue", "RTE": "glue", "WNLI": "glue",
                     "ANLI": "anli"}

TASKS_TO_TASKS = {"CoLA": "cola", "SST-2": "sst2", "MRPC": "mrpc", "STS-B": "stsb", "QQP": "qqp",
                  "MNLI-m": "mnli", "MNLI-mm": "mnli", "QNLI": "qnli", "RTE": "rte", "WNLI": "wnli",
                  "ANLI": None}

TASKS_TO_DATASETS_KEY = {"CoLA": ("train", "validation", "test"),
                         "SST-2": ("train", "validation", "test"),
                         "MRPC": ("train", "validation", "test"),
                         "STS-B": ("train", "validation", "test"),
                         "QQP": ("train", "validation", "test"),
                         "MNLI-m": ("train", "validation_matched", "test_matched"),
                         "MNLI-mm": ("train", "validation_mismatched", "test_mismatched"),
                         "QNLI": ("train", "validation", "test"), 
                         "RTE": ("train", "validation", "test"), 
                         "WNLI": ("train", "validation", "test"),
                         "ANLI": ("train"+anli_part, "dev"+anli_part, "test"+anli_part)}

TASKS_TO_KEYS = {"CoLA": ("sentence", None), "SST-2": ("sentence", None), "MRPC": ("sentence1", "sentence2"),
                 "STS-B": ("sentence1", "sentence2"), "QQP": ("question1", "question2"), "MNLI-m": ("premise", "hypothesis"),
                 "MNLI-mm": ("premise", "hypothesis"), "QNLI": ("question", "sentence"), "RTE": ("sentence1", "sentence2"),
                 "WNLI": ("sentence1", "sentence2"), "ANLI": ("premise", "hypothesis")}

TASKS_TO_LABELS = {"CoLA": ("0", "1"), "SST-2": ("0", "1"), "MRPC": ("0", "1"), "QQP": ("0", "1"),
                   "MNLI-m": ("entailment", "neutral", "contradiction"), "MNLI-mm": ("entailment", "neutral", "contradiction"),
                   "QNLI": ("entailment", "not_entailment"), "RTE": ("entailment", "not_entailment"),
                   "WNLI": ("0", "1"), "ANLI": ("0", "1", "2")}

DATASETS_TO_METRICS = {"glue":"glue", "anli":"accuracy"}

def LoadDataset(task_name):
    # 加载数据集
    print("-" * 16, "load the dataset", "-" * 16)
    print("[Notice]: loading dataset...")
    dataset_name = TASKS_TO_DATASETS[task_name]
    config_name = TASKS_TO_TASKS[task_name]
    metric_name = DATASETS_TO_METRICS[dataset_name]
    raw_dataset = load_dataset("datasets/" + dataset_name, config_name)
    metric = load_metric("metrics/" + metric_name, config_name)
    print("[Notice]: dataset of", task_name, "is loaded.")
    print("-" * 50)
    return raw_dataset, metric

def LoadModel(task_name, model_name):
    # 加载tokenizer和model
    print("-" * 8, "load the tokenizer and the model", "-" * 8)
    print("[Notice]: loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    num_labels = len(TASKS_TO_LABELS[task_name]) if task_name != "STS-B" else 1
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    print("[Notice]: tokenizer and model are loaded.")
    print("-" * 50)
    return tokenizer, model

def Tokenize(task_name, tokenizer, raw_dataset):
    print("-" * 14, "tokenize the dataset", "-" * 14)
    print("[Notice]: tokenizing the dataset...")
    # Tokenize文本
    def preprocess_function(examples):
        sentence1_key, sentence2_key = TASKS_TO_KEYS[task_name]
        texts = ((examples[sentence1_key],) if sentence2_key is None else (
            examples[sentence1_key], examples[sentence2_key]))
        result = tokenizer(*texts, max_length=token_max_length, truncation=True)
        result["labels"] = examples["label"]
        return result
    preprocess_function = preprocess_function
    old_columns = raw_dataset["train"].column_names if task_name!="ANLI" else raw_dataset["train_r1"].column_names
    tokenized_dataset = raw_dataset.map(preprocess_function, batched=True, remove_columns=old_columns, keep_in_memory=True)
    print("[Notice]: the dataset is tokenized.")
    print("-" * 50)
    return tokenized_dataset

def MakeDataloader(task_name, tokenizer, tokenized_dataset, batch_size):
    # 设置训练集、验证集、测试集
    print("-" * 14, "make the dataloader", "-" * 15)
    print("[Notice]: making dataloader...")
    train_key, validation_key, test_key = TASKS_TO_DATASETS_KEY[task_name]
    train_dataset = tokenized_dataset[train_key]
    val_dataset = tokenized_dataset[validation_key]
    test_dataset = tokenized_dataset[test_key]
    # 创建dataloader
    data_collator = DataCollatorWithPadding(tokenizer)
    train_dataloader = DataLoader(train_dataset, collate_fn=data_collator, shuffle=True, batch_size=batch_size)
    eval_dataloader = DataLoader(val_dataset, collate_fn=data_collator, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=batch_size)
    print("[Notice]: the dataloader is made.")
    print("-" * 50)
    return (train_dataloader, eval_dataloader, test_dataloader)

def preprocess(task_name, model_name, batch_size):
    raw_dataset, metric = LoadDataset(task_name)
    tokenizer, model = LoadModel(task_name, model_name)
    tokenized_dataset = Tokenize(task_name, tokenizer, raw_dataset)
    dataloader = MakeDataloader(task_name, tokenizer, tokenized_dataset, batch_size)
    return model, dataloader, metric