import pandas as pd
from datasets import Dataset

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import (
    CrossEncoderTrainingArguments,
)

# Загружаем данные
df_wiki = pd.read_csv(
    "data/rupaws_wiki_train.csv", sep=";", names=["sentence1", "sentence2", "label"]
)

df_qqp = pd.read_csv(
    "../data/rupaws_qqp_train.csv", sep=";", names=["sentence1", "sentence2", "label"]
)

df = pd.concat([df_wiki, df_qqp])

# Dataset HF
train_dataset = Dataset.from_pandas(df)

# Модель
model = CrossEncoder(
    "DiTy/cross-encoder-russian-msmarco",
    num_labels=1,
)

# Аргументы обучения
args = CrossEncoderTrainingArguments(
    output_dir="./artifacts/cross_encoder_finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=8,
)

# Trainer
trainer = CrossEncoderTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
)

trainer.train()

model.save("./artifacts/cross_encoder_finetuned")
