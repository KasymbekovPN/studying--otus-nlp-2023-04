import torch
import pandas as pd
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5Tokenizer, T5ForConditionalGeneration, \
    get_scheduler, get_linear_schedule_with_warmup

PRETRAINED_PATH = "ai-forever/ruT5-base"


class EvalDataset(Dataset):
    def __init__(self, text, tokenizer, length, device):
        self._text = text.reset_index(drop=True)
        self._tokenizer = tokenizer
        self._length = length
        self._device = device

    def __len__(self):
        return self._text.shape[0]

    def __getitem__(self, item):
        output = self._tokenize(self._text[item])
        return {k: v.reshape(-1).to(self._device) for k, v in output.items()}

    def _tokenize(self, text):
        return self._tokenizer(text,
                               return_tensors='pt',
                               padding='max_length',
                               truncation=True,
                               max_length=self._length)


class TrainDataset(Dataset):
    POS_LABEL = 'верно'
    NEG_LABEL = 'неверно'

    def __init__(self, text, label, tokenizer, length, device):
        self._text = text.reset_index(drop=True)
        self._label = label.reset_index(drop=True)
        self._tokenizer = tokenizer
        self._length = length
        self._device = device

    def __len__(self):
        return self._label.shape[0]

    def __getitem__(self, item):
        output = self._tokenize(self._text[item], self._length)
        output = {k: v.reshape(-1).to(self._device) for k, v in output.items()}

        label = self.POS_LABEL if self._label[item] == 1 else self.NEG_LABEL
        label = self._tokenize(label, length=2).input_ids.reshape(-1).to(self._device)

        output.update({'labels': label})
        return output

    def _tokenize(self, text, length):
        return self._tokenizer(text,
                               return_tensors='pt',
                               padding='max_length',
                               truncation=True,
                               max_length=length)


def eval_model(model, dataloader):
    model.eval()
    # loss = []
    # for batch in dataloader:
    #     outputs = model(**batch)
    #     loss.append(outputs.loss.item())
    loss = [model(**batch).loss.item() for batch in dataloader]
    model.train()
    return np.sum(loss) / len(loss)


def train_model(model,
                train_dataloader,
                eval_dataloader,
                optimizer,
                scheduler,
                n_epochs,
                train_ds_len):
    model.train()
    for epoch in range(n_epochs):
        print(f'EPOCH {epoch + 1} of {n_epochs}')
        for batch_id, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            if batch_id % 50 == 0:
                loss_train = loss.item()
                current = batch_id * batch['input_ids'].shape[0]
                print(f'\tTrain loss: {loss_train} [{current}/{train_ds_len}]')
                print("Evaluating...")
                loss_val = eval_model(model, eval_dataloader)
                print(f'Eval loss: {loss_val}')
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()


def test_model(model, dataloader, pos_label):
    model.eval()
    result = np.array([])
    for batch in dataloader:
        tokens = model.generate(**batch)
        tokens = [1 if pos_label in token else 0 for token in tokens]
        result = np.hstack([result, tokens])

    return result


def run():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_PATH, use_fast=False)
    model = T5ForConditionalGeneration.from_pretrained(PRETRAINED_PATH)
    model.to(device)

    # x = tokenizer(TrainDataset.POS_LABEL,
    #               return_tensors='pt',
    #               padding='max_length',
    #               truncation=True,
    #               max_length=2)
    # print(x)
    # print(tokenizer.decode(2937))
    # print(2937 in x['input_ids'])
    # print(tokenizer.)
    #             gen_tok = [1 if 2937 in i else 0 for i in gen_tok]  # tokenizer.decode(2937) == 'верно'

    train_eval_df = pd.read_csv('../../hw_004_bert_gpt3_t5_practice/train_dataset.csv', usecols=[1, 2])
    idx = train_eval_df.sample(frac=0.9, random_state=42).index
    val_df = train_eval_df[~train_eval_df.index.isin(idx)]
    train_df = train_eval_df[train_eval_df.index.isin(idx)]
    test_df = pd.read_csv('../../hw_004_bert_gpt3_t5_practice/test_dataset.csv', usecols=[1, 2])

    batch_size = 32

    train_ds = TrainDataset(train_df['sentence'], train_df['acceptable'], tokenizer, 64, device)
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    eval_ds = TrainDataset(val_df['sentence'], val_df['acceptable'], tokenizer, 64, device)
    eval_dataloader = DataLoader(eval_ds, batch_size=32)

    # rename EvalDataset
    test_ds = EvalDataset(test_df['sentence'], tokenizer, 64, device)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size)

    optimizer = Adam(model.parameters(), lr=1e-5)

    num_epochs = 2
    num_training_steps = num_epochs * len(train_dataloader)
    scheduler = get_scheduler('linear', optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    train_model(model, train_dataloader, eval_dataloader, optimizer, scheduler, num_epochs, len(train_ds))

    pos_label = tokenizer(TrainDataset.POS_LABEL,
                          return_tensors='pt',
                          padding='max_length',
                          truncation=True,
                          max_length=2)['input_ids'][0][0].item()
    y_prediction = test_model(model, test_dataloader, pos_label)

    print(f'F1 score: {f1_score(y_prediction, test_df["acceptable"])}')

    # train_model(train_dataloader, num_epochs)
    # *-------------------
    # y_pred = test_model(test_dataloader, eval=False)

    # print(f'F1-score = {f1_score(y_pred, test["acceptable"]):>3f}\n')

    pass


if __name__ == '__main__':
    run()
