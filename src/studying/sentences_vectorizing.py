# https://habr.com/ru/articles/515036/
import datetime
import os.path
import re

import gensim.models
import nltk
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer


class WordsOnly:
    def __init__(self, pattern):
        self._pattern = pattern

    def __call__(self, *args, **kwargs):
        try:
            return ' '.join(self._pattern.findall(args[0]))
        except:
            return ''


class Lemmatizator:
    DOWNLOAD_TASKS = ['punkt', 'wordnet']

    def __init__(self):
        [nltk.download(task) for task in self.DOWNLOAD_TASKS]

    def __call__(self, *args, **kwargs):
        tokens = nltk.word_tokenize(args[0])
        wnl = WordNetLemmatizer()
        return ' '.join([wnl.lemmatize(word, 'v') for word in tokens])


def get_model():
    model_path = './sentences_vectorizing.model'
    if os.path.isfile(model_path):
        return gensim.models.Word2Vec.load(model_path)

    # !!!
    dataset_path = '../hw_003_vector/jupyter/IMDB Dataset.csv'
    frame = pd.read_csv(dataset_path, sep=',', usecols=[0])

    words_only = WordsOnly(pattern=re.compile("[А-Яа-я:=!)(A-z_%/|]+"))
    lem = Lemmatizator()

    texts = [lem(words_only(row['review'])).split() for idx, row in frame.iterrows()]

    m = gensim.models.Word2Vec(texts, window=5, min_count=5, vector_size=300)
    m.save(model_path)

    return m


def get_vector(token: str, vectors):
    return vectors.vectors[vectors.key_to_index[token]] if token in vectors.key_to_index else None


class Vectors:
    def __init__(self, wv):
        self._wv = wv

    def get(self, token: str):
        return self._wv.vectors[self._wv.key_to_index[token]] if token in self._wv.key_to_index else None


def avg_embedding(vectors: Vectors, sentence: str):
    sentence_vector = []
    for token in sentence.split():
        sentence_vector.append(vectors.get(token))
    v = np.mean(sentence_vector, axis=0)
    print(v.shape)
    print(v)


def tfidf_embedding(vectors: Vectors, sentence: str, vectorizer):
    weighs_data1 = vectorizer.transform([sentence])
    weighs_data = vectorizer.transform([sentence]).tocoo()

    vocab = vectorizer.get_feature_names_out()

    sentence_vector = []
    for row, col, weight in zip(weighs_data.row, weighs_data.col, weighs_data.data):
        print(row, col, weight)
        print(vocab[col])
        token = vectors.get(vocab[col])
        if token is not None:
            sentence_vector.append(weight * token)
    v = np.mean(sentence_vector, axis=0)
    print(v.shape)
    print(v)

    pass

# def similarity_values_wrapper(use_word2vec=True, use_fasttext=True, distance_function=cosine_distances):
#     def get_similarity_values(sentences):
#         sent_vector = [[]]*len(sentences)
#         weights_data = tf_idf_vectorizer.transform(sentences).tocoo()
#         for row, col, weight in zip(weights_data.row, weights_data.col, weights_data.data):
#             sent_vector[row].append(weight*vectorize(vocab[col], use_word2vec, use_fasttext))
#
#         for row in range(len(sent_vector)):
#             if not sent_vector[row]:
#                 sent_vector.append((len(vectorize('zoros_vector'))))
#         sent_vector = np.sum(sent_vector, axis=1)
#         distances = distance_function(sent_vector, sent_vector)
#         return distances
#     return get_similarity_values




if __name__ == '__main__':
    # m = gensim.models.Word2Vec.load('../../datasets/enwiki_20180420_nolg_300d.txt')

    # print(datetime.datetime.now())
    # m = gensim.models.keyedvectors.load_word2vec_format('../../datasets/enwiki_20180420_nolg_300d.txt')
    # print(type(m))
    # print(datetime.datetime.now())

    model = get_model()

    # sentence = 'Hello world I am code'.split()
    # for word in sentence:
    #     v = get_vector(word, model.wv)
    #     print(v)

    vs = Vectors(model.wv)
    x = avg_embedding(vs, 'Hello world')

    dataset_path = '../hw_003_vector/jupyter/IMDB Dataset.csv'
    frame = pd.read_csv(dataset_path, sep=',', usecols=[0])
    corpus = [row['review'] for index, row in frame.iterrows()]

    tf_idf_vectorizer = TfidfVectorizer()
    tf_idf_vectorizer.fit(corpus)
    # vocab = tf_idf_vectorizer.get_feature_names_out()

    x = tfidf_embedding(vs, 'Hello world I am Oz', tf_idf_vectorizer)


    pass