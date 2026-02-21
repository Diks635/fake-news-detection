import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)


df = pd.read_csv("data/news.csv")

df = df.sample(2000, random_state=42)


df["label"] = df["label"].map({"FAKE": 0, "REAL": 1})


df = df.dropna(subset=["text", "label"])


train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)


tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)

train_encodings = tokenizer(
    train_texts,
    truncation=True,
    padding=True,
    max_length=128  
)

val_encodings = tokenizer(
    val_texts,
    truncation=True,
    padding=True,
    max_length=128
)


class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = NewsDataset(train_encodings, train_labels)
val_dataset = NewsDataset(val_encodings, val_labels)


model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


training_args = TrainingArguments(
    output_dir="model/distilbert_fake_news",
    per_device_train_batch_size=8,   
    num_train_epochs=1,              
    weight_decay=0.01,
    save_strategy="no",
    logging_steps=100,
    report_to="none"
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator
)


trainer.train()

model.save_pretrained("model/distilbert_fake_news")
tokenizer.save_pretrained("model/distilbert_fake_news")

print("✅ DistilBERT Fake News Model Trained Successfully")
