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
        return {k: v.reshape(-1).to(self._device) for k, v in output.items() }

    def _tokenize(self, text):
        return self._tokenizer(text,
                               return_tensors='pt',
                               paddind='max_length',
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
        label = self._tokenize(label, length=2).input_ids.respape(-1).to(self._device)

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
        print(f'EPOCH {epoch+1} of {n_epochs}')
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

# < !!!
#       pos_label - gen_tok = [1 if 2937 in i else 0 for i in gen_tok]  # tokenizer.decode(2937) == 'верно'
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

    # train_ds = TrainDataset(train['sentence'], train['acceptable'])
    # train_dataloader = DataLoader(train_ds, batch_size=32, shuffle=True)
    #
    # eval_ds = TrainDataset(val['sentence'], val['acceptable'])
    # eval_dataloader = DataLoader(eval_ds, batch_size=32)
    #
    # test_ds = EvalDataset(test['sentence'])
    # test_dataloader = DataLoader(test_ds, batch_size=32)

    # -----------------------

    # optimizer = Adam(model.parameters(), lr=1e-5)
    #
    # num_epochs = 2
    # num_training_steps = num_epochs * len(train_dataloader)
    # lr_scheduler = get_scheduler(
    #     "linear",
    #     optimizer=optimizer,
    #     num_warmup_steps=0,
    #     num_training_steps=num_training_steps
    # )

    # -------------------------

    # --------------------------

    # train_model(train_dataloader, num_epochs)

    # *--------------------

    # y_pred = test_model(test_dataloader, eval=False)
    # print(f'F1-score = {f1_score(y_pred, test["acceptable"]):>3f}\n')

    pass


if __name__ == '__main__':
    run()
