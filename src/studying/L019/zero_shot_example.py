import torch
import re

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskGeneration, GPT2ForSequenceClassification


def define_device():
    if torch.cuda.is_available():
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print(f'We will use the GPU: {torch.cuda.get_device_name(0)}.')
        device = torch.device('cuda')
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device('cpu')
    return device


def calc_loss(phrase: str,
              tokenizer,
              model,
              device):
    phrase = tokenizer.encode(phrase)
    if len(phrase) == 1:
        phrase.append(tokenizer.eos_token_id)
    phrase = torch.tensor(phrase, dtype=torch.long, device=device)
    phrase = phrase.unsqueeze(0)  # .repeat(num_samples, 1)
    with torch.no_grad():
        loss = model(phrase, labels=phrase)
    loss[0].item() # ???
    return loss[0].item()


def calc_loss_(phrase: str,
               tokenizer,
               model,
               device):
    phrase = tokenizer.encode(phrase)
    if len(phrase) == 1:
        phrase.append(tokenizer.eos_token_id)
    phrase = torch.tensor(phrase, dtype=torch.long, device=device)
    phrase = phrase.unsqueeze(0)  # .repeat(num_samples, 1)
    with torch.no_grad():
        # loss = model(phrase, labels=phrase)
        loss = model(phrase)
    print(loss)
    # loss[0].item() # ???
    # return loss[0].item()


def clean(text: str):
    text = re.sub(r'\((\d+)\)', '', text)
    return text


def run():
    device = define_device()
    print(f'device: {device}')

    pretrained_name = 'ai-forever/rugpt3large_based_on_gpt2'
    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
    model = AutoModelForCausalLM.from_pretrained(pretrained_name)

    g_model = GPT2ForSequenceClassification.from_pretrained(pretrained_name)

    model.cuda()
    g_model.cuda()

    def get_loss_num_(text):
        loss = calc_loss_(text, tokenizer, g_model, device)
        print(f'loss: {loss}: text: {text}')

    def get_loss_num(text):
        loss = calc_loss(text, tokenizer, model, device)
        print(f'loss: {loss}: text: {text}')

    # text = 'жизнь отличная'
    #
    # get_loss_num('Позитивный твит: ' + text)
    # get_loss_num('Позитивный твит: ' + text + ')')
    # get_loss_num('Позитивный твит: ' + text + ' )')
    # get_loss_num('Позитивный твит: ' + text + ')))')
    # get_loss_num('Позитивный твит: ' + text + ' )))')
    #
    # get_loss_num('Негативный твит: ' + text)
    # get_loss_num('Негативный твит: ' + text + '(')
    # get_loss_num('Негативный твит: ' + text + ' (')
    # get_loss_num('Негативный твит: ' + text + '(((')
    # get_loss_num('Негативный твит: ' + text + ' (((')
    #
    # print()
    # text = 'Кот сидел на коврике.'
    # few_shots = ['Предложение далее корректное? ' + 'Мама мыла раму.' + " Ответ: да.",
    #              'Предложение далее корректное? ' + 'Пингвить идить туды.' + " Ответ: нет."]
    # get_loss_num('\n'.join(few_shots) + '\nПредложение далее корректное? ' + text + " Ответ: да.")
    #
    # print()
    # text = 'Кот сидеть на коврике.'
    # get_loss_num('\n'.join(few_shots) + 'Предложение далее корректное? ' + text + " Ответ: нет.")

    # get_loss_num_('Негативный твит: ' + text + ' (((')

    text = 'Позитивный твит: жизнь прекрасна !!!'
    inputs = tokenizer(text, return_tensors='pt').to(device)
    with torch.no_grad():
        logits = g_model(**inputs).logits.to('cpu')

    predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]
    print(predicted_class_ids)

    text = 'Негативный твит: жизнь прекрасна !!!'
    inputs = tokenizer(text, return_tensors='pt').to(device)
    with torch.no_grad():
        logits = g_model(**inputs).logits.to('cpu')

    predicted_class_ids1 = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]
    print(predicted_class_ids1)


if __name__ == '__main__':
    run()
