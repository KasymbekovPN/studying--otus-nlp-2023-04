import torch
import pandas as pd

from sklearn.metrics import f1_score
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

tqdm.pandas()


def run():

    test_df = pd.read_csv('../../hw_004_bert_gpt3_t5_practice/test_dataset.csv', usecols=[1, 2])

    # train = pd.read_csv("/content/RuCoLA/data/in_domain_train.csv", usecols=[1, 2])
    # test = pd.read_csv("/content/RuCoLA/data/in_domain_dev.csv", usecols=[1, 2])


    pretrained_model_path = 'ai-forever/rugpt3large_based_on_gpt2'
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_path)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Device: {device}')
    model.to(device)

    def calc_loss(text):
        inputs = tokenizer.encode(text, return_tensors='pt').reshape(-1).to(device)
        with torch.no_grad():
            loss = model(input_ids=inputs, labels=inputs).loss.item()
        return loss

    print(calc_loss('Предложение корректно?'))
    print(calc_loss('Предложение корректно ?'))
    print(calc_loss('Привет, мир'))
    print(calc_loss('Привет '))
    print(calc_loss('Привет'))

    # zero

    def shot(start: str, text: str, end: list):
        first = ' '.join([start, text, end[0]])
        second = ' '.join([start, text, end[1]])

        loss_1, loss_2 = calc_loss(first), calc_loss(second)
        return 1 if loss_1 > loss_2 else 0

    y_pred = test_df['sentence'].progress_apply(lambda x: shot('Проверь корректность предложения.', x, ['Это предложение корректное.', 'Это предложение некорректно']))

    print(f'F1-score = {f1_score(y_pred, test_df["acceptable"]):>3f}\n')


    y_pred = test_df['sentence'].progress_apply(lambda x: shot('Если ли здесь ошибка?', x, ['Предложение правильное.', 'Допущена ошибка']))
    print(f'F1-score = {f1_score(y_pred, test_df["acceptable"]):>3f}\n')

    # few shot
    promt = """Проверь корректность предложения:
    Вдруг решетка беззвучно поехала в сторону, и на балконе возникла таинственная фигура, прячущаяся от лунного света, и погрозила Ивану пальцем. => Верно
    Этим летом не никуда ездили. => Неверно    
    """
    y_pred = test_df['sentence'].apply(lambda x: shot(promt, x, [' => Верно', ' => Неверно']))
    print(f'F1-score = {f1_score(y_pred, test_df["acceptable"]):>3f}\n')

# ---------    "Few shot"

    #     "# 2 shots\n",
    #     "promt = \"\"\"Проверь корректность предложения:\n",
    #     "Вдруг решетка беззвучно поехала в сторону, и на балконе возникла таинственная фигура, прячущаяся от лунного света, и погрозила Ивану пальцем. => Верно\n",
    #     "Этим летом не никуда ездили. => Неверно\n",
    #     "\"\"\"\n",
    #     "y_pred = test['sentence'].apply(lambda x: shot(promt, x, ['=> Верно', '=> Неверно']))\n",
    #     "print(f'F1-score = {f1_score(y_pred, test[\"acceptable\"]):>3f}\\n')"



    #     "# 2 shots. Change promt\n",
    #     "promt = \"\"\"Проверь корректность предложения:\n",
    #     "Вдруг решетка беззвучно поехала в сторону, и на балконе возникла таинственная фигура, прячущаяся от лунного света, и погрозила Ивану пальцем. Предложение правильное.\n",
    #     "Этим летом не никуда ездили. Допущена ошибка\n",
    #     "\"\"\"\n",
    #     "y_pred = test['sentence'].apply(lambda x: shot(promt, x, ['Предложение правильное.', 'Допущена ошибка']))\n",
    #     "print(f'F1-score = {f1_score(y_pred, test[\"acceptable\"]):>3f}\\n')"


    #     "# 4 shots\n",
    #     "promt = \"\"\"Проверь корректность предложения:\n",
    #     "Вдруг решетка беззвучно поехала в сторону, и на балконе возникла таинственная фигура, прячущаяся от лунного света, и погрозила Ивану пальцем. Предложение правильное.\n",
    #     "Этим летом не никуда ездили. Допущена ошибка\n",
    #     "На поверку вся теория оказалась полной чепухой. Предложение правильное.\n",
    #     "Симптомов болезни не исчезло. Допущена ошибка\n",
    #     "\"\"\"\n",
    #     "y_pred = test['sentence'].apply(lambda x: shot(promt, x, ['Предложение правильное.', 'Допущена ошибка']))\n",
    #     "print(f'F1-score = {f1_score(y_pred, test[\"acceptable\"]):>3f}\\n')"






if __name__ == '__main__':
    run()
