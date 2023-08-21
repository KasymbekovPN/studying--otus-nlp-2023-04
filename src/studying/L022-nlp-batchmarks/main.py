import os
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from functools import partial
from shutil import rmtree

from datasets import load_metric
from razdel import tokenize
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BertForSequenceClassification,
    BertTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    set_seed,

    T5Tokenizer,
    T5ForConditionalGeneration

)

from datasets import Dataset, DatasetDict

ACCURACY = load_metric("accuracy", keep_in_memory=True)
MCC = load_metric("matthews_correlation", keep_in_memory=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

N_SEEDS = 10
N_EPOCHS = 5
LR_VALUES = (1e-5, 3e-5, 5e-5)
DECAY_VALUES = (1e-4, 1e-2, 0.1)
BATCH_SIZES = (32, 64)

DATA_DIR = './data/'
TRAIN_FILE = DATA_DIR + "in_domain_train.csv"
IN_DOMAIN_DEV_FILE = DATA_DIR + "in_domain_dev.csv"
OUT_OF_DOMAIN_DEV_FILE = DATA_DIR + "out_of_domain_dev.csv"
TEST_FILE = DATA_DIR + "test.csv"

MODEL_NAME = 'ai-forever/ruBert-base'


def compute_metrics(p: EvalPrediction):
    preds = p.predictions
    preds = np.argmax(preds, axis=1)

    acc_result = ACCURACY.compute(predictions=preds, references=p.label_ids)
    mcc_result = MCC.compute(predictions=preds, references=p.label_ids)

    result = {"accuracy": acc_result["accuracy"], "mcc": mcc_result["matthews_correlation"]}

    return result


def preprocess_examples(examples, tokenizer):
    result = tokenizer(examples["sentence"], padding=False)

    if "acceptable" in examples:
        result["label"] = examples["acceptable"]

    result["length"] = [len(list(tokenize(sentence))) for sentence in examples["sentence"]]
    return result


def read_splits(*, as_datasets):
    train_df, in_domain_dev_df, out_of_domain_dev_df, test_df = map(
        pd.read_csv, (TRAIN_FILE, IN_DOMAIN_DEV_FILE, OUT_OF_DOMAIN_DEV_FILE, TEST_FILE)
    )

    # concatenate datasets to get aggregate metrics
    dev_df = pd.concat((in_domain_dev_df, out_of_domain_dev_df))

    if as_datasets:
        train, dev, test = map(Dataset.from_pandas, (train_df, dev_df, test_df))
        return DatasetDict(train=train, dev=dev, test=test)
    else:
        return train_df, dev_df, test_df


def run():
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    splits = read_splits(as_datasets=True)

    tokenized_splits = splits.map(
        partial(preprocess_examples, tokenizer=tokenizer),
        batched=True,
        remove_columns=["sentence"],
        keep_in_memory=True,
    )

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    # seed, lr, wd, bs
    dev_metrics_per_run = np.empty((N_SEEDS, len(LR_VALUES), len(DECAY_VALUES), len(BATCH_SIZES), 2))

    best_mcc = -float("inf")

    for i, learning_rate in enumerate(LR_VALUES):
        for j, weight_decay in enumerate(DECAY_VALUES):
            for k, batch_size in enumerate(BATCH_SIZES):
                for seed in range(N_SEEDS):
                    set_seed(seed)

                    model = BertForSequenceClassification.from_pretrained(MODEL_NAME)

                    training_args = TrainingArguments(
                        output_dir=f"checkpoints/",
                        overwrite_output_dir=True,
                        evaluation_strategy="epoch",
                        per_device_train_batch_size=batch_size,
                        per_device_eval_batch_size=batch_size,
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                        num_train_epochs=N_EPOCHS,
                        warmup_ratio=0.1,
                        optim="adamw_torch",
                        save_strategy="epoch",
                        save_total_limit=1,
                        seed=seed,
                        dataloader_num_workers=4,
                        group_by_length=True,
                        report_to="none",
                        load_best_model_at_end=True,
                        metric_for_best_model="eval_mcc",
                    )

                    trainer = Trainer(
                        model=model,
                        args=training_args,
                        train_dataset=tokenized_splits["train"],
                        eval_dataset=tokenized_splits["dev"],
                        compute_metrics=compute_metrics,
                        tokenizer=tokenizer,
                        data_collator=data_collator,
                    )

                    train_result = trainer.train()
                    print("train", train_result.metrics)

                    dev_predictions = trainer.predict(test_dataset=tokenized_splits["dev"])
                    print("dev", dev_predictions.metrics)
                    dev_metrics_per_run[seed, i, j, k] = (
                        dev_predictions.metrics["test_accuracy"],
                        dev_predictions.metrics["test_mcc"],
                    )

                    predictions = trainer.predict(test_dataset=tokenized_splits["test"])

                    if dev_predictions.metrics["test_mcc"] > best_mcc:
                        print(f"Found new best model! {learning_rate} {weight_decay} {batch_size} {seed}")
                        best_mcc = dev_predictions.metrics["test_mcc"]


if __name__ == '__main__':
    run()
