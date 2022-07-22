from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForMultipleChoice, default_data_collator

max_seq_length = 256
AnswerToLabel = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, '': -1}

def LoadDataset(task_name):
    # 加载数据集
    print("-" * 16, "load the dataset", "-" * 16)
    print("[Notice]: loading dataset...")
    raw_dataset = load_dataset("datasets/" + task_name)
    metric = load_metric("metrics/" + "accuracy")
    print("[Notice]: dataset", task_name, "is loaded.")
    print("-" * 50)
    return raw_dataset, metric

def LoadModel(model_name):
    # 加载tokenizer和model
    print("-" * 8, "load the tokenizer and the model", "-" * 8)
    print("[Notice]: loading tokenizer and model...")
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMultipleChoice.from_pretrained(model_name, config=config)
    print("[Notice]: tokenizer and model are loaded.")
    print("-" * 50)
    return tokenizer, model

def Tokenize(tokenizer, raw_dataset):
    print("-" * 14, "tokenize the dataset", "-" * 14)
    print("[Notice]: tokenizing the dataset...")
    # Tokenize文本
    def preprocess_function(examples):
        sentences = []
        for choice in examples["choices"]['text']:
            sentences.append([examples["question"],choice])
        labels = AnswerToLabel[examples['answerKey']] if AnswerToLabel[examples['answerKey']]!='' else '-1'
        tokenized_inputs = tokenizer(sentences, max_length=max_seq_length, padding="max_length", truncation=True)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    tokenized_dataset = raw_dataset.map(preprocess_function, remove_columns=raw_dataset["train"].column_names, keep_in_memory=True)
    print("[Notice]: the dataset is tokenized.")
    print("-" * 50)
    return tokenized_dataset

def MakeDataloader(tokenized_dataset, batch_size):
    # 设置训练集、验证集、测试集
    print("-" * 14, "make the dataloader", "-" * 15)
    print("[Notice]: making dataloader...")
    train_dataset = tokenized_dataset["train"]
    val_dataset = tokenized_dataset["validation"]
    test_dataset = tokenized_dataset["test"]
    # 创建dataloader
    data_collator = default_data_collator
    train_dataloader = DataLoader(train_dataset, collate_fn=data_collator, shuffle=True, batch_size=batch_size)
    eval_dataloader = DataLoader(val_dataset, collate_fn=data_collator, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=batch_size)
    print("[Notice]: the dataloader is made.")
    print("-" * 50)
    return (train_dataloader, eval_dataloader, test_dataloader)

def preprocess(task_name, model_name, batch_size):
    raw_dataset, metric = LoadDataset(task_name)
    tokenizer, model = LoadModel(model_name)
    tokenized_dataset = Tokenize(tokenizer, raw_dataset)
    dataloader = MakeDataloader(tokenized_dataset, batch_size)
    return model, dataloader, metric