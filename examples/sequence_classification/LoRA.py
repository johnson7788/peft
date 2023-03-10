#!/usr/bin/env python
# coding: utf-8
import argparse
import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PeftModel,
    PeftConfig,
)

import evaluate
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from tqdm import tqdm


def train_model(num_epochs=2, use_lora=True):
    batch_size = 32
    model_name_or_path = "roberta-large"
    task = "mrpc"  #任务类型
    peft_type = PeftType.LORA
    device = "cuda"
    peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)
    lr = 3e-4
    if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
        padding_side = "left" #如果模型名称或路径中包含"gpt"、"opt"或"bloom"等字符串，则填充在左侧
    else:
        padding_side = "right"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
    if getattr(tokenizer, "pad_token_id") is None:
        # 如果tokenizer没有pad_token_id属性，则设置为eos_token_id
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def tokenize_function(examples):
        # max_length=None => 使用模型的最大长度（实际上是默认设置）
        outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=None)
        return outputs

    def collate_fn(examples):
        # 将数据进行padding，并将其转化为tensor
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    datasets = load_dataset("glue", task)
    metric = evaluate.load("glue", task)
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "sentence1", "sentence2"],
    )
    # 将"label"列重命名为"labels"，这是transformers库中模型的预期标签名称
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    #  实例化dataloader。
    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn,
                                  batch_size=batch_size)
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], shuffle=False, collate_fn=collate_fn, batch_size=batch_size
    )
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True)
    if use_lora:
        model = get_peft_model(model, peft_config)
        print(f"打印模型中可训练的参数")
        model.print_trainable_parameters()
    # 打印模型中可训练的参数
    print(f"打印模型结构")
    print(model)
    optimizer = AdamW(params=model.parameters(), lr=lr)

    # Instantiate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0.06 * (len(train_dataloader) * num_epochs),
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch.to(device)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        model.eval()
        for step, batch in enumerate(tqdm(eval_dataloader)):
            batch.to(device)
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = predictions, batch["labels"]
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()
        print(f"epoch {epoch}:", eval_metric)
    # 保存模型到本地
    model.save_pretrained("roberta-large-peft-lora")
    print(f"保存模型成功")
    # 模型文件包括,adapter_config.json, adapter_model.bin, 模型很小
    # -rw-rw-r-- 1 johnson johnson  349 Mar 10 12:51 adapter_config.json
    # -rw-rw-r-- 1 johnson johnson 7.1M Mar 10 12:51 adapter_model.bin

# ## Share adapters on the 🤗 Hub

# model.push_to_hub("smangrul/roberta-large-peft-lora", use_auth_token=True)
# ## Load adapters from the Hub
# You can also directly load adapters from the Hub using the commands below:

def model_test(device,eval_dataloader,metric):
    peft_model_id = "smangrul/roberta-large-peft-lora"
    config = PeftConfig.from_pretrained(peft_model_id)
    inference_model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    # Load the Lora model
    inference_model = PeftModel.from_pretrained(inference_model, peft_model_id)

    inference_model.to(device)
    inference_model.eval()
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch.to(device)
        with torch.no_grad():
            outputs = inference_model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = predictions, batch["labels"]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )

    eval_metric = metric.compute()
    print(eval_metric)


if __name__ == '__main__':
    train_model()


