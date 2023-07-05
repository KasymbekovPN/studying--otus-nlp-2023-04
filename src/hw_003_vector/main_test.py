import datetime

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import json
import json
import nltk
import re
import seaborn
import warnings
import random

import pymorphy2
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.pipeline import *
from sklearn.metrics import *
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict
from nltk import FreqDist
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import *
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import Normalizer, LabelEncoder
from gensim.models import Word2Vec
from gensim.models import *
from gensim import corpora
from gensim import similarities

from bokeh.models import ColumnDataSource, LabelSet
from bokeh.plotting import figure, show, output_file
from bokeh.io import output_notebook
from scipy.cluster.hierarchy import ward, dendrogram
from pymystem3 import Mystem
from matplotlib import style

warnings.filterwarnings("ignore")
random.seed(1228)
pd.set_option('display.max_colwidth', None)
style.use('ggplot')


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


def label2num(label: str) -> int:
    if label == 'positive':
        return 1
    elif label == 'negative':
        return -1
    return 0


def num2label(y: str) -> str:
    if y == 1:
        return 'positive'
    elif y == -1:
        return 'negative'
    return 'neutral'


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

    # text = 'в этот вечер мы слушаем вебинар по обработке естественного языка в отус'
    # stemmed_text = ' '.join([morph.parse(x)[0].normal_form for x in text.split(' ')])
    # print(f'Original text: {text}')
    # print(f'Lemmatized text: {stemmed_text}')


# bag of words
def demo1():
    with open('../../datasets/kazakh_news/train.json', encoding='utf-8') as json_file:
        data = json.load(json_file)

    sentiments = [datum['sentiment'] for datum in data]
    encoded_sentiments = [label2num(label) for label in sentiments]

    path = '../../datasets/kazakh_news/text_lemmatized.txt'
    texts = [line.replace('\n', '') for line in open(path, encoding='utf-8').readlines()]

    idx = 1
    print(f'sentiment: {sentiments[idx]}')
    print(f'text: {texts[idx]}')

    train_texts, test_texts, train_y, test_y = train_test_split(texts,
                                                                encoded_sentiments,
                                                                test_size=0.2,
                                                                random_state=42,
                                                                stratify=sentiments)

    vectorizer = CountVectorizer(max_features=100)
    vectorizer.fit(train_texts)

    print(vectorizer.get_feature_names_out()[:10])

    train_X = vectorizer.transform(train_texts)
    test_X = vectorizer.transform(test_texts)
    # print(train_X.todense()[:2])
    # print(type(train_X))

    # train_X = vectorizer.transform(train_texts)
    # train_X.todense()[:2]

    clf = RandomForestClassifier(n_estimators=500, max_depth=10)
    clf = clf.fit(train_X, train_y)
    pred = clf.predict(test_X)

    print(f'pred: {pred[:20]}')
    print(f'test_y: {test_y[:20]}')

    decoded_pred = [num2label(y) for y in pred]
    decoded_test_y = [num2label(y) for y in test_y]
    print(f'Pred labels: {decoded_pred[:20]}')
    print(f'Original labels: {decoded_test_y[:20]}')

    print(f'Accuracy: {accuracy_score(test_y, pred)}')
    print(f'F1: {f1_score(test_y, pred, average="macro")}')


# tf-idf
def demo2():
    with open('../../datasets/kazakh_news/train.json', encoding='utf-8') as json_file:
        data = json.load(json_file)

    sentiments = [datum['sentiment'] for datum in data]
    encoded_sentiments = [label2num(label) for label in sentiments]

    path = '../../datasets/kazakh_news/text_lemmatized.txt'
    texts = [line.replace('\n', '') for line in open(path, encoding='utf-8').readlines()]

    idx = 1
    print(f'sentiment: {sentiments[idx]}')
    print(f'text: {texts[idx]}')

    print(f'texts.size: {len(texts)}')
    train_texts, test_texts, train_y, test_y = train_test_split(texts,
                                                                encoded_sentiments,
                                                                test_size=0.2,
                                                                random_state=42,
                                                                stratify=sentiments)

    vectorizer = TfidfVectorizer(max_features=200, norm=None)
    vectorizer.fit(train_texts)
    print(vectorizer.get_feature_names_out()[:20])

    train_X = vectorizer.fit_transform(train_texts)
    test_X = vectorizer.transform(test_texts)
    # print(train_X.todense()[:2])

    clf = RandomForestClassifier(n_estimators=500, max_features=10)
    clf.fit(train_X, train_y)
    pred = clf.predict(test_X)

    decoded_pred = [num2label(y) for y in pred]
    decoded_test_y = [num2label(y) for y in test_y]
    print('Предсказанные метки: ', decoded_pred[0:20], ".....")
    print('Истинные метки: ', decoded_test_y[0:20], ".....")

    print(f'Accuracy: {accuracy_score(test_y, pred)}')
    print(f'F1: {f1_score(test_y, pred, average="macro")}')


# 10 vector representation
def demo3():
    m = Mystem()
    regex = re.compile("[А-Яа-я:=!\)\()A-z\_\%/|]+")

    def words_only(text, regex=regex):
        try:
            return ' '.join(regex.findall(text))
        except:
            return ''

    def lemmatize(text, mystem=m):
        try:
            return ''.join(m.lemmatize(text)).strip()
        except:
            ' '

    text0 = 'g;iuhoikl 7.kjh 87h одлжд :))'
    print(words_only(text0))

    df_pos = pd.read_csv('../../datasets/twitter/positive.csv', sep=';', header=None, usecols=[3])
    print('\nPositive tail')
    print(df_pos.tail(3))

    df_neg = pd.read_csv('../../datasets/twitter/negative.csv', sep=';', header=None, usecols=[3])
    print('\nNegative head')
    print(df_neg.head(3))

    # df_neg = pd.read_csv("negative.csv", sep=';', header=None, usecols=[3])
    # df_pos = pd.read_csv("positive.csv", sep=';', header=None, usecols=[3])

    df_neg['sent'] = 'neg'
    df_pos['sent'] = 'pos'
    df_neg['text'] = df_neg[3]
    df_pos['text'] = df_pos[3]

    df = pd.concat([df_neg, df_pos])
    df = df[['text', 'sent']]

    df.text = df.text.apply(words_only)

    df = pd.read_csv('../../datasets/twitter/processed_text.csv', index_col=0)
    print('\nProcessed text')
    print(df.head())
    print(f'df.shape: {df.shape}')

    texts = [df.text.iloc[i].split() for i in range(len(df))]

    model = Word2Vec(texts, window=5, min_count=5, workers=4, vector_size=200)
    model.save('../../datasets/twitter/word2v.model')


# 10 vector representation
def demo4():
    model = Word2Vec.load('../../datasets/twitter/word2v.model')

    print('model.wv.most_similar("школа")')
    print(model.wv.most_similar("школа"))

    print('model.wv.most_similar("school")')
    print(model.wv.most_similar("school"))

    print('model.wv.most_similar("работа")')
    print(model.wv.most_similar("работа"))

    print("vec = (model.wv['университет'] - model.wv['студент'] + model.wv['школьник']) / 3")
    vec = (model.wv['университет'] - model.wv['студент'] + model.wv['школьник']) / 3
    print(model.wv.similar_by_vector(vec))

    print('model.wv.doesnt_match("ночь улица фонарь аптека".split())')
    print(model.wv.doesnt_match("ночь улица фонарь аптека".split()))

    df = pd.read_csv('../../datasets/twitter/processed_text.csv', index_col=0)
    print('\nProcessed text')
    print(df.head())
    print(f'df.shape: {df.shape}')

    texts = [df.text.iloc[i].split() for i in range(len(df))]

    top_words = []
    fd = FreqDist()
    for text in texts:
        fd.update(text)
    for i in fd.most_common(500):
        top_words.append(i[0])
    print(top_words)

    # model.tra

    print(' ---- ', model.wv['школа'])
    # print(' ++++ ', model.wv.key_to_index)
    # print(' ++++ ', model.wv.index_to_key)

    top_words_vec = model.wv[top_words]
    print(f'top_words_vec.shape {top_words_vec.shape}')
    # print(f'top_words_vec[0]: {top_words_vec[0]}')

    tsne = TSNE(n_components=2, random_state=0)
    top_words_tsne = tsne.fit_transform(top_words_vec)

    # output_notebook()
    output_file('../../output/bakeh.html')
    p = figure(tools='pan,wheel_zoom,reset,save',
               toolbar_location="above",
               title='word2vec T-SNE for most common words')
    source = ColumnDataSource(data=dict(x1=top_words_tsne[:, 0],
                                        x2=top_words_tsne[:, 1],
                                        names=top_words))
    p.scatter(x='x1', y='x2', size=8, source=source)
    labels = LabelSet(x='x1', y='x2', text='names', y_offset=6, text_font_size='8pt',
                      text_color='#555555', source=source, text_align='center')
    p.add_layout(labels)
    # show(p)

    dist = 1 - cosine_similarity(top_words_vec)
    linkage_matrix = ward(dist)

    fig, ax = plt.subplots(figsize=(10, 100))
    ax = dendrogram(linkage_matrix, orientation='right', labels=top_words)
    plt.tick_params(axis='x',
                    which='both',
                    bottom='off',
                    top='off',
                    labelbottom='off')
    plt.tight_layout()
    plt.savefig('../../output/image.png', dpi=200)

    pass


# 13 topic modeling
def demo5():
    stop_words = load_stop_words()

    # df = pd.read_csv('../../datasets/twitter/processed_text.csv', index_col=0)
    # texts = [row['text'] for idx, row in df.iterrows()]
    # content = ''
    # delimiter = ''
    # for i in range(25_000):
    #     content += delimiter + texts[i]
    #     delimiter = '\n\n'
    # with open('../../output/topic_text.txt', 'w', encoding='utf-8') as f:
    #     f.write(content)

    with open('../../output/topic_text.txt', 'r', encoding='utf-8') as f:
        texts = f.read().split('\n\n')

    print(len(texts))
    print(texts[0])

    punctuation_re = r'[!"#$%&()*+,./:;<=>?@[\]^_`{|}~„“«»†*/\—–‘’]'
    numbers_re = r'[0-9]'

    texts = [re.sub('\n', ' ', text) for text in texts]
    texts = [re.sub(punctuation_re, '', text) for text in texts]
    texts = [re.sub(numbers_re, '', text) for text in texts]

    tokenized_texts = []
    for text in texts:
        text = [w for w in text.split() if w not in stop_words]
        tokenized_texts.append(text)

    print(len(tokenized_texts))

    print('Making dictionary...')
    dictionary = corpora.Dictionary(tokenized_texts)
    print(f'Original: {dictionary}')

    dictionary.filter_extremes(no_below=5, no_above=0.9, keep_n=None)
    dictionary.save('../../output/some.dict')
    print(f'Filtered: {dictionary}')

    print('Vectorizing corpus...')
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    corpora.MmCorpus.serialize('../../output/some.model', corpus)

    print(len(tokenized_texts), len(corpus))

    tfidf = TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    sampling_tfidf = random.choices(corpus_tfidf, k=30)
    print(f'sampling_tfidf: {sampling_tfidf}')

    index = similarities.MatrixSimilarity(sampling_tfidf)
    print(f'index: {index}')

    sims = index[sampling_tfidf]
    # print(f'sims: {sims}')

    plt.figure(figsize=(12, 10))
    seaborn.heatmap(data=sims, cmap='Spectral').set(xticklabels=[], yticklabels=[])
    plt.title('Sim matrix')
    plt.show()

    pass


if __name__ == '__main__':
    # demo0()
    # demo1()
    # demo2()
    # demo3()
    demo4()
    # demo5()
