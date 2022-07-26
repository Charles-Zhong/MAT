import torch
from itertools import chain
from dataclasses import dataclass
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForMultipleChoice, PreTrainedTokenizerBase

@dataclass
class DataCollatorForMultipleChoice: # https://huggingface.co/docs/transformers/tasks/multiple_choice
    tokenizer: PreTrainedTokenizerBase
    def __call__(self, features):
        label_name = "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [[{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features]
        flattened_features = list(chain(*flattened_features))
        batch = self.tokenizer.pad(flattened_features,return_tensors="pt")
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch

AnswerToLabel = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, '1': 0, '2': 1, '3': 2, '4': 3, '': -1}

def LoadDataset(dataset_name, task_name):
    # 加载数据集
    print("-" * 16, "load the dataset", "-" * 16)
    print("[Notice]: loading dataset...")
    raw_dataset = load_dataset("datasets/" + dataset_name, task_name)
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
        while len(examples['choices']['text']) < 5:
            examples['choices']['text'].append('')
            examples['choices']['label'].append('')
        sentences = []
        for choice in examples["choices"]['text']:
            sentences.append([examples["question"], choice])
        tokenized_inputs = tokenizer(sentences)
        tokenized_inputs["labels"] = AnswerToLabel[examples['answerKey']]
        return tokenized_inputs
    tokenized_dataset = raw_dataset.map(preprocess_function, remove_columns=raw_dataset["train"].column_names, keep_in_memory=True)
    print("[Notice]: the dataset is tokenized.")
    print("-" * 50)
    return tokenized_dataset

def MakeDataloader(tokenizer, tokenized_dataset, batch_size):
    # 设置训练集、验证集、测试集
    print("-" * 14, "make the dataloader", "-" * 15)
    print("[Notice]: making dataloader...")
    train_dataset = tokenized_dataset["train"]
    val_dataset = tokenized_dataset["validation"]
    test_dataset = tokenized_dataset["test"]
    # 创建dataloader
    data_collator = DataCollatorForMultipleChoice(tokenizer)
    train_dataloader = DataLoader(train_dataset, collate_fn=data_collator, shuffle=True, batch_size=batch_size)
    eval_dataloader = DataLoader(val_dataset, collate_fn=data_collator, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=batch_size)
    print("[Notice]: the dataloader is made.")
    print("-" * 50)
    return (train_dataloader, eval_dataloader, test_dataloader)

def preprocess(dataset_name, task_name, model_name, batch_size):
    raw_dataset, metric = LoadDataset(dataset_name, task_name)
    tokenizer, model = LoadModel(model_name)
    tokenized_dataset = Tokenize(tokenizer, raw_dataset)
    dataloader = MakeDataloader(tokenizer, tokenized_dataset, batch_size)
    return model, dataloader, metric