import torch
import pandas as pd
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5Tokenizer, T5ForConditionalGeneration, \
    get_scheduler, get_linear_schedule_with_warmup

# --------------

# tokenizer = AutoTokenizer.from_pretrained("ai-forever/ruT5-base", use_fast=False)
# model = T5ForConditionalGeneration.from_pretrained("ai-forever/ruT5-base")

# ----------------

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# model.to(device);

# ------------------

# class EvalDataset(Dataset):
#
#     def __init__(self, X):
#         self.text = X.reset_index(drop=True)
#
#     def tokenize(self, text):
#         return tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=45)
#
#     def __len__(self):
#         return self.text.shape[0]
#
#     def __getitem__(self, index):
#         output = self.text[index]
#         output = self.tokenize(output)
#         return {k: v.reshape(-1).to(device) for k, v in output.items()}
#
#
# class TrainDataset(Dataset):
#
#     def __init__(self, X, label):
#         self.text = X.reset_index(drop=True)
#         self.label = label.reset_index(drop=True)
#
#     def tokenize(self, text, length=45):
#         return tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=length)
#
#     def __len__(self):
#         return self.label.shape[0]
#
#     def __getitem__(self, index):
#         output = self.text[index]
#         output = self.tokenize(output)
#         output = {k: v.reshape(-1).to(device) for k, v in output.items()}
#
#         label = 'верно' if self.label[index] == 1 else 'неверно'
#         label = self.tokenize(label, length=2).input_ids.reshape(-1).to(device)
#
#         output.update({'labels': label})
#         return output
#
#
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

# def train_model(train_dataloader, num_epochs):
#     model.train()
#     for epoch in range(num_epochs):
#         print(f'Epoch {epoch+1} \n -------------------')
#         for n_batch, batch in enumerate(train_dataloader):
#             outputs = model(**batch)
#             loss = outputs.loss
#             if n_batch % 50 == 0:
#                 loss_train, current = loss.item(), n_batch * batch['input_ids'].shape[0]
#                 print(f"Loss train: {loss_train:>7f}  [{current:>5d}/{len(train_ds):>5d}]")
#                 print('Evaluating...')
#                 loss_val, _ = test_model(eval_dataloader, eval=True)
#                 print(f"Loss test: {loss_val:>7f}\n")
#             loss.backward()
#             optimizer.step()
#             lr_scheduler.step()
#             optimizer.zero_grad()
#
# def test_model(test_dataloader, eval=False):
#     model.eval()
#     y_pred = np.array([])
#     y_true = np.array([])
#     loss = []
#     for n_batch, batch in enumerate(test_dataloader):
#         if not eval:
#             gen_tok = model.generate(**batch)
#             gen_tok = [1 if 2937 in i else 0 for i in gen_tok]  # tokenizer.decode(2937) == 'верно'
#             y_true = np.hstack([y_true, gen_tok])
#         else:
#             outputs = model(**batch)
#             loss.append(outputs.loss.item())
#     if not eval:
#         return y_true
#     else:
#         return np.sum(loss)/len(loss), y_true

# --------------------------

# train_model(train_dataloader, num_epochs)

# *--------------------


# y_pred = test_model(test_dataloader, eval=False)
# print(f'F1-score = {f1_score(y_pred, test["acceptable"]):>3f}\n')

def run():
    pass


if __name__ == '__main__':
    run()
