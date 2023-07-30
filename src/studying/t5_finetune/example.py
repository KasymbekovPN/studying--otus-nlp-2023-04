import os
import pathlib

import torch
import wget
import numpy as np
import pandas as pd

from pathlib import Path
from functools import partial
from shutil import rmtree
from datasets import load_metric, Dataset, DatasetDict
from razdel import tokenize
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, T5Tokenizer, \
    T5ForConditionalGeneration

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_NAME = 't5-base'
MODEL_TO_HUB_NAME = {
    't5-base': 'ai-forever/ruT5-base'
}

ACCURACY = load_metric('accuracy', keep_in_memory=True)
MCC = load_metric('matthews_correlation', keep_in_memory=True)

N_SEEDS = 1
N_EPOCHS = 20
LR_VALUES = (1e-4, 1e-3)
DECAY_VALUES = (0, 1e-4)
BATCH_SIZES = (128,)

POS_LABEL = 'yes'
NEG_LABEL = 'no'

TRAIN_FILE_URL = 'https://github.com/RussianNLP/RuCoLA/blob/main/data/in_domain_train.csv?raw=true'
IN_DOMAIN_DEV_URL = 'https://github.com/RussianNLP/RuCoLA/blob/main/data/in_domain_dev.csv?raw=true'
OUT_OF_DOMAIN_DEV_URL = 'https://github.com/RussianNLP/RuCoLA/blob/main/data/out_of_domain_dev.csv?raw=true'
TEST_FILE_URL = 'https://github.com/RussianNLP/RuCoLA/blob/main/data/test.csv?raw=true'

CURRENT_DIR = Path(__file__).parent
DATA_DIR = CURRENT_DIR / 'data'

TRAIN_FILE = DATA_DIR / 'in_domain_train.csv'
IN_DOMAIN_DEV_FILE = DATA_DIR / 'in_domain_dev.csv'
OUT_OF_DOMAIN_DEV_FILE = DATA_DIR / 'out_of_domain_dev.csv'
TEST_FILE = DATA_DIR / 'test.csv'

ARGS = {
    'overwrite_output_dir': True,
    'evaluation_strategy': 'epoch',
    'lr_scheduler_type': 'constant',
    'save_strategy': 'epoch',
    'save_total_limit': 1,
    'fp16': True,
    'dataloader_num_workers': 4,
    'group_by_length': True,
    'report_to': 'none',
    'load_best_model_at_end': True,
    'metric_for_best_model': 'eval_mcc',
    'optim': 'adafactor',
    'predict_with_generate': True
}


def check_or_create_directory(path: Path) -> None:
    if not os.path.isdir(path):
        pathlib.Path.mkdir(pathlib.Path(path))


def load_dataset(url: str, file_path: Path) -> None:
    file_path = str(file_path)
    if os.path.exists(file_path):
        print(f'Dataset {file_path} is already downloaded.')
    else:
        wget.download(url, file_path)
        print(f'Dataset {file_path} is downloaded.')


def read_splits():
    train_df, in_domain_dev_df, out_of_domain_dev_df, test_df = map(
        pd.read_csv,
        (TRAIN_FILE, IN_DOMAIN_DEV_FILE, OUT_OF_DOMAIN_DEV_FILE, TEST_FILE)
    )

    # concatenate datasets to get aggregate metrics
    dev_df = pd.concat((in_domain_dev_df, out_of_domain_dev_df))

    train, dev, test = map(Dataset.from_pandas, (train_df, dev_df, test_df))
    return DatasetDict(train=train, dev=dev, test=test)


def compute_metrics(p, tokenizer):
    string_preds = tokenizer.batch_decode(p.predictions, skip_special_tokens=True)
    int_preds = [1 if prediction == POS_LABEL else 0 for prediction in string_preds]

    labels = np.where(p.label_ids != -100, p.label_ids, tokenizer.pad_token_id)
    string_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    int_labels = []

    for string_label in string_labels:
        if string_label == POS_LABEL:
            int_labels.append(1)
        elif string_label == NEG_LABEL or string_label == '': # second case accounts for test data
            int_labels.append(0)
        else:
            raise ValueError()

    acc_result = ACCURACY.compute(predictions=int_preds, references=int_labels)
    mcc_result = MCC.compute(predictions=int_preds, references=int_labels)

    result = {'accuracy': acc_result['accuracy'], 'mcc': mcc_result['matthews_correlation']}
    return result


def preprocess_examples(examples, tokenizer):
    result = tokenizer(examples['sentence'], padding=True)

    if 'acceptable' in examples:
        label_sequences = []
        for label in examples['acceptable']:
            if label == 1:
                target_sequence = POS_LABEL
            elif label == 0:
                target_sequence = NEG_LABEL
            else:
                raise ValueError('Unknown class label')
            label_sequences.append(target_sequence)
    else:
        label_sequences = ['' for _ in examples['sentence']]

    result['labels'] = tokenizer(label_sequences, padding=False)['input_ids']
    result['length'] = [len(list(tokenize(sequence))) for sequence in examples['sentence']]

    return result


def create_training_args(run_base_dir,
                         batch_size,
                         learning_rate,
                         weight_decay,
                         seed):
    return Seq2SeqTrainingArguments(
        output_dir=f'checkpoints/{run_base_dir}',
        overwrite_output_dir=ARGS['overwrite_output_dir'],
        evaluation_strategy=ARGS['evaluation_strategy'],
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_train_epochs=N_EPOCHS,
        lr_scheduler_type=ARGS['lr_scheduler_type'],
        save_strategy=ARGS['save_strategy'],
        save_total_limit=ARGS['save_total_limit'],
        seed=seed,
        fp16=ARGS['fp16'],
        dataloader_num_workers=ARGS['dataloader_num_workers'],
        group_by_length=ARGS['group_by_length'],
        report_to=ARGS['report_to'],
        load_best_model_at_end=ARGS['load_best_model_at_end'],
        metric_for_best_model=ARGS['metric_for_best_model'],
        optim=ARGS['optim'],
        predict_with_generate=ARGS['predict_with_generate']
    )


def run():
    check_or_create_directory(DATA_DIR)

    load_dataset(TRAIN_FILE_URL, TRAIN_FILE)
    load_dataset(IN_DOMAIN_DEV_URL, IN_DOMAIN_DEV_FILE)
    load_dataset(OUT_OF_DOMAIN_DEV_URL, OUT_OF_DOMAIN_DEV_FILE)
    load_dataset(TEST_FILE_URL, TEST_FILE)

    tokenizer = T5Tokenizer.from_pretrained(MODEL_TO_HUB_NAME[MODEL_NAME])
    splits = read_splits()

    tokenized_splits = splits.map(
        partial(preprocess_examples, tokenizer=tokenizer),
        batched=True,
        remove_columns=['sentence']
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)
    dev_metrics_per_run = np.empty((N_SEEDS, len(LR_VALUES), len(DECAY_VALUES), len(BATCH_SIZES), 2))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    for i, learning_rate in enumerate(LR_VALUES):
        for j, weight_decay in enumerate(DECAY_VALUES):
            for k, batch_size in enumerate(BATCH_SIZES):
                for seed in range(N_SEEDS):
                    model = T5ForConditionalGeneration.from_pretrained(MODEL_TO_HUB_NAME[MODEL_NAME])
                    model.to(device)
                    run_base_dir = f'{MODEL_NAME}_{learning_rate}_{weight_decay}_{batch_size}'

                    training_args = create_training_args(run_base_dir, batch_size, learning_rate, weight_decay, seed)
                    trainer = Seq2SeqTrainer(
                        model=model,
                        args=training_args,
                        train_dataset=tokenized_splits['train'],
                        eval_dataset=tokenized_splits['dev'],
                        compute_metrics=partial(compute_metrics, tokenizer=tokenizer),
                        tokenizer=tokenizer,
                        data_collator=data_collator
                    )

                    train_result = trainer.train()
                    print(f'{run_base_dir}_{seed}')
                    print(f'train {train_result.metrics}')

                    os.makedirs(f'results/{run_base_dir}_{seed}', exist_ok=True)

                    dev_predictions = trainer.predict(
                        test_dataset=tokenized_splits['dev'],
                        metric_key_prefix='test',
                        max_length=10
                    )
                    print(f'dev {dev_predictions.metrics}')
                    dev_metrics_per_run[seed, i, j, k] = (
                        dev_predictions.metrics['test_accuracy'],
                        dev_predictions.metrics['test_mcc']
                    )

                    predictions = trainer.predict(test_dataset=tokenized_splits['test'], max_length=10)
                    string_preds = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)

                    int_preds = [1 if predictions == POS_LABEL else 0 for prediction in string_preds]
                    int_preds = np.asarray(int_preds)

                    np.save(f'results/{run_base_dir}_{seed}/preds.npy', int_preds)
                    rmtree(f'checkpoints/{run_base_dir}')

    os.makedirs('result_agg', exist_ok=True)
    np.save(f'result_agg/{MODEL_NAME}_dev.npy', dev_metrics_per_run)


if __name__ == '__main__':
    run()
