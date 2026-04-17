"""
DistilBERT 文本分类训练脚本
用法: python train.py --data train.csv --output ./best_model
"""

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)


class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="train.csv")
    parser.add_argument("--output", type=str, default="./best_model")
    parser.add_argument(
        "--model_name", type=str, default="distilbert-base-multilingual-cased"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    # 加载数据
    df = pd.read_csv(args.data)
    print(
        f"数据集: {len(df)} 条, "
        f"正常: {(df['label']==0).sum()}, "
        f"敏感: {(df['label']==1).sum()}"
    )

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["text"].tolist(),
        df["label"].tolist(),
        test_size=args.test_size,
        random_state=42,
        stratify=df["label"],
    )

    # 分词
    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_name)
    train_encodings = tokenizer(
        train_texts, truncation=True, padding=True, max_length=args.max_length
    )
    val_encodings = tokenizer(
        val_texts, truncation=True, padding=True, max_length=args.max_length
    )

    train_dataset = TextDataset(train_encodings, train_labels)
    val_dataset = TextDataset(val_encodings, val_labels)

    # 模型
    model = DistilBertForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    )

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # 保存
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"模型已保存到 {args.output}")

    # 最终评估
    results = trainer.evaluate()
    print(f"验证集结果: {results}")


if __name__ == "__main__":
    main()
