import os
import zipfile
import time
import datetime
import random
import wget
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import matthews_corrcoef
from torch.utils.data import random_split, TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig, \
    get_linear_schedule_with_warmup


def print_header(h: str):
    print(f'===== {h.upper()} =====')


def run():
    print_header('check cuda')
    if torch.cuda.is_available():
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print(f'We will use the GPU: {torch.cuda.get_device_name(0)}')
        device = torch.device('cuda')
    else:
        print('No GPU available, using the GPU instead.')
        device = torch.device('cpu')
    print(f'device: {device}')

    print_header('downloading dataset...')
    url = 'https://nyu-mll.github.io/CoLA/cola_public_1.1.zip'
    zip_path = './dataset/cola_public_1.1.zip'
    if os.path.exists(zip_path):
        print(f'{zip_path} is downloaded')
    else:
        wget.download(url, zip_path)

    print_header('unpacking dataset')
    if not os.path.exists('./dataset/cola_public/'):
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            zip_file.extractall('./dataset/')
            print('Dataset extracted')
    else:
        print('Dataset already has been extracted')

    print_header('load the dataset into a pandas dataframe')
    df = pd.read_csv(
        './dataset/cola_public/raw/in_domain_train.tsv',
        delimiter='\t',
        header=None,
        names=['sequence_source', 'label', 'label_notes', 'sentence']
    )
    print(f'Number of training sentences: {df.shape[0]}')

    r_size = 10
    print_header(f'display {r_size} random rows from the data')
    sample = df.sample(10)
    print(sample)

    r_size = 5
    print_header(f'display {r_size} rows from the data')
    sample = df.loc[df.label == 0].sample(r_size)[['sentence', 'label']]
    print(sample)

    print_header('get the lists of sentences and their labels')
    sentences = df.sentence.values
    labels = df.label.values

    print_header('Loading bert tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    print(tokenizer)

    idx = 0
    print_header(f'print sentence {idx}')
    print(f'Original: {sentences[0]}')
    print(f'Tokenized: {tokenizer.tokenize(sentences[0])}')
    print(f'Token IDs: {tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences[0]))}')

    print_header('Define max length')
    max_len = 0
    for sentence in sentences:
        input_ids = tokenizer.encode(sentence, add_special_tokens=True)
        max_len = max(max_len, len(input_ids))
    print(f'Max sentence length: {max_len}')

    print_header('tokenize all of the sentences')
    input_ids = []
    attention_masks = []
    for sentence in sentences:
        encoded_dict = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=64,
            # pad_to_max_length=True,
            # padding=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict["attention_mask"])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    print(f'Original: {sentences[idx]}')
    print(f'Token IDs: {input_ids[idx]}')

    print_header('Combine the training inputs into a TensorDataset')
    dataset = TensorDataset(input_ids, attention_masks, labels)

    print_header('create 90/10 train/validation split')
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    print(f'training samples: {train_size}, validation samples: {val_size}')
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print_header('set batch size')
    batch_size = 32
    print(f'Batch size = {batch_size}')

    print_header('Training & validation dataloaders creation')
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size
    )
    validation_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=batch_size
    )

    print('create bert-model')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False
    )
    model.cuda()

    print_header('get all of the model\'s parameters as a list of tuples')
    params = list(model.named_parameters())
    print('The BERT model has {:} different named parameters.\n'.format(len(params)))
    print('=== Embedding layer ===')
    for p in params[0:5]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    print('\n=== First Transformer ===\n')
    for p in params[5:21]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    print('\n=== Output Layer ===\n')
    for p in params[-4:]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print_header('create optimizer')
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

    print_header('set epochs amount')
    # epochs = 4
    epochs = 2
    print(f'Epochs: {epochs}')

    print_header('calculate total steps amount')
    total_steps = len(train_dataloader) * epochs
    print(f'Total steps amount: {total_steps}')

    print_header('create the learning rate scheduler')
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    print_header('create function to calculation of accuracy of predictions vs labels')

    def flat_accuracy(predictions, labels):
        predictions_flat = np.argmax(predictions, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(predictions_flat == labels_flat) / len(labels_flat)

    def format_time(elapsed):
        elapsed_rounded = int(round(elapsed))
        return str(datetime.timedelta(seconds=elapsed_rounded))

    print_header('fix seed')
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)

    print_header('train/valid')
    training_stats = []
    total_t0 = time.time()

    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        t0 = time.time()
        total_train_loss = 0

        model.train()

        for step, batch in enumerate(train_dataloader):
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()

            res = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels
            )
            loss = res['loss']
            logits = res['logits']

            total_train_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

        print("")
        print("Running Validation...")

        t0 = time.time()
        model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        for batch in validation_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            with torch.no_grad():
                res = model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels
                )
            loss = res['loss']
            logits = res['logits']
            total_eval_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            total_eval_accuracy += flat_accuracy(logits, label_ids)

        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        avg_val_loss = total_eval_loss / len(validation_dataloader)
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

    print_header('display stats')
    df_stats = pd.DataFrame(data=training_stats)
    df_stats = df_stats.set_index('epoch')
    print(df_stats)

    print('Display in plot')
    sns.set(style='darkgrid')
    sns.set(font_scale=1.5)
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
    plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")

    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks([1, 2, 3, 4])

    plt.show()

    print_header('load cola dataset into a pandas dataframe')
    df = pd.read_csv(
        './dataset/cola_public/raw/out_of_domain_dev.tsv',
        delimiter='\t',
        header=None,
        names=['sentence_source', 'label', 'label_notes', 'sentence']
    )
    print(df.head())
    print('Number of test sentences: {:,}\n'.format(df.shape[0]))

    print_header('Create sentence and label lists')
    sentences = df.sentence.values
    labels = df.label.values

    print_header('Tokenize all of the sentences and map the tokens to thier word IDs.')

    input_ids = []
    attention_masks = []
    for sentence in sentences:
        encoded_dict = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=64,
            # pad_to_max_length=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    print_header('Convert the lists into tensors.')
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    print_header('Set the batch size.')
    batch_size = 32
    print(f'batch size: {batch_size}')

    print_header('Create the DataLoader')
    prediction_data = TensorDataset(input_ids, attention_masks, labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    print_header('Prediction on test set')
    print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))

    model.eval()
    predictions, true_labels = [], []

    for barch in prediction_dataloader:
        batch = tuple(t.to(device) for t in barch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask
            )
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        predictions.append(logits)
        true_labels.append(label_ids)
    print('    DONE.')
    print('Positive samples: %d of %d (%.2f%%)' % (df.label.sum(), len(df.label), (df.label.sum() / len(df.label) * 100.0)))

    print_header('Evaluate each test batch using Matthew\'s correlation coefficient')
    matthews_set = []
    for i in range(len(true_labels)):
        pred_labels_i = np.argmax(predictions[i], axis=1).flatten()
        matthews = matthews_corrcoef(true_labels[i], pred_labels_i)
        matthews_set.append(matthews)

    print_header('Combine the results across all batches.')
    flat_predictions = np.concatenate(predictions, axis=0)
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    flat_true_labels = np.concatenate(true_labels, axis=0)
    mcc = matthews_corrcoef(flat_true_labels, flat_predictions)
    print('Total MCC: %.3f' % mcc)

    output_dir = './dataset/save/'
    print_header(f'saving model to {output_dir}')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


# # Load a trained model and vocabulary that you have fine-tuned
# model = BertForSequenceClassification.from_pretrained(output_dir)
# tokenizer = BertTokenizer.from_pretrained(output_dir)
#
# # Copy the model to the GPU.
# model.to(device)

if __name__ == '__main__':
    run()
