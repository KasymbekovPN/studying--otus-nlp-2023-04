import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import json
import nltk
import re

import pymorphy2
from tqdm import tqdm
from sklearn.metrics import *
from collections import Counter
from nltk.stem.snowball import SnowballStemmer

import warnings

warnings.filterwarnings("ignore")


def get_sentiment_set(data: list) -> set:
    return set(datum['sentiment'] for datum in data)


def print_datum(idx, data):
    print(f'ID: {data[idx]["id"]}')
    print(f'Text: {data[idx]["text"]}')
    print(f'Sentiment: {data[idx]["sentiment"]}')


def count_sentiments(data):
    return Counter([datum['sentiment'] for datum in data])


def load_stop_words():
    # nltk.download('stopwords')
    return nltk.corpus.stopwords.words('russian')


def tokenize(line: str) -> list:
    return nltk.WordPunctTokenizer().tokenize(line)


def get_words_only(line: str) -> str:
    regex = re.compile(r'[А-Яа-яA-zёЁ-]+')
    try:
        return ' '.join(regex.findall(line)).lower()
    except:
        return ''


def process_data(data, stop_words):
    texts = []

    for item in tqdm(data):
        tokens = tokenize(get_words_only(item['text']))
        tokens = [word for word in tokens if word not in stop_words and not word.isnumeric()]
        texts.append(tokens)

    return texts


def demo0() -> None:
    with open('../../datasets/kazakh_news/train.json', encoding='utf-8') as json_file:
        data = json.load(json_file)

    sentiments = get_sentiment_set(data)
    # print('sentiments: ', sentiments)

    # print_datum(1, data)

    counters = count_sentiments(data)
    print(f'Counters: {counters}')

    stop_words = load_stop_words()

    # tokens = tokenize('казнить, нельзя помиловать!!!')
    # print(f'tokens: {tokens}')
    #
    # tokens = tokenize(get_words_only('Казнить, нельзя помиловать!!!'))
    # print(f'tokens: {tokens}')

    add_stop_words = ['kz', 'казахстан', 'астана', 'казахский', 'алматы', 'ао', 'оао', 'ооо']
    months = ['январь', 'февраль', 'март', 'апрель', 'май', 'июнь', 'июль', 'август', 'сентябрь', 'октябрь', 'ноябрь',
              'декабрь', ]
    all_stop_words = stop_words + add_stop_words + months

    texts = process_data(data, all_stop_words)

    # example
    i = 1
    sentiments = [datum['sentiment'] for datum in data]
    print(f'label: {sentiments[i]}')
    print(f'Tokens: {texts[i]}')

    # stemmer = SnowballStemmer('russian')
    # for word in texts[i][:10]:
    #     sword = stemmer.stem(word)
    #     print(f'before: {word}, after: {sword}')

    morph = pymorphy2.MorphAnalyzer()
    # print(morph.parse('студентам'))
    # print(morph.parse('лук'))

    for word in texts[i][:10]:
        norm = morph.parse(word)[0].normal_form
        print(f'original: {word}, norm: {norm}')

    pass


if __name__ == '__main__':
    demo0()
