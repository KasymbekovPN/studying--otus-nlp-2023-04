import copy
import csv
import datetime
import gensim

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *

from src.hw_003_vector.configurator.configurator import Configurator
from src.hw_003_vector.reader.dataset_reader import DatasetReader
from src.hw_003_vector.preparation.preparation import Preparation
from src.hw_003_vector.preparation.case_preparator import CasePreparator
from src.hw_003_vector.preparation.regexp_preparators import PunctuationPreparator, SpacePreparator, HtmlTagsPreparator
from src.hw_003_vector.preparation.text_lemm_preparator import TextLemmPreparator
from src.hw_003_vector.preparation.sentiment_read_preparator import SentimentReadPreparator
from src.hw_003_vector.preparation.sentiment_write_preparator import SentimentWritePreparator
from src.hw_003_vector.dataset.sentiment_strategies import convert_sentiment_to_str, convert_sentiment_to_int
from src.hw_003_vector.dataset.dataset import Dataset


def write_prepared_dataset(dataset: list, conf: Configurator):
    with open(conf('path.dataset.prepared'), 'w', newline='', encoding='utf-8') as file:
        fieldnames = ['review', 'sentiment']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        [writer.writerow(datum) for datum in dataset]


def get_dataset(conf: Configurator) -> Dataset | None:
    dataset_reader = DatasetReader()
    dataset_reader(path_original=conf('path.dataset.original'), path_prepared=conf('path.dataset.prepared'))

    saver = write_prepared_dataset
    if dataset_reader.prepared is not None:
        result = dataset_reader.prepared
        saver = None
        read_preparator = Preparation(SentimentReadPreparator())
    elif dataset_reader.original is None:
        print('No one dataset')
        return None
    else:
        result = dataset_reader.original
        read_preparator = Preparation(
            CasePreparator(),
            HtmlTagsPreparator(),
            PunctuationPreparator(),
            SpacePreparator(),
            TextLemmPreparator(),
            SentimentReadPreparator()
        )

    result = [read_preparator(datum) for datum in result]
    if saver is not None:
        prepared = copy.deepcopy(result)
        write_preparator = Preparation(SentimentWritePreparator())
        prepared = [write_preparator(datum) for datum in prepared]
        saver(prepared, conf)

    return Dataset(result,
                   {'review', 'sentiment'},
                   strategy_int_sentiment=convert_sentiment_to_int,
                   strategy_str_sentiment=convert_sentiment_to_str)


def print_timediff(tag, prev_time):
    current = datetime.datetime.now()
    print(f'[{tag}]: {current - prev_time}')

    return current


def run():
    conf = Configurator()

    dataset = get_dataset(conf)
    if dataset is None:
        return

    current_time = datetime.datetime.now()
    train_x, test_x, train_y, test_y = train_test_split(
        dataset.get_column('review'),
        dataset.get_column('sentiment', 'int_sentiment'),
        test_size=conf('train-test-split.test-size'),
        random_state=conf('train-test-split.random-stage'),
        stratify=dataset.get_column('sentiment', 'str_sentiment')
    )
    current_time = print_timediff('slice', current_time)

    vectorizer = TfidfVectorizer(
        max_features=conf('vec.tfidf.max-features'),
        norm=conf('vec.tfidf.norm'),
        max_df=conf('vec.tfidf.max-df'),
        min_df=conf('vec.tfidf.min-df'),
        stop_words=conf('vec.tfidf.stop-words')
    )
    train_X = vectorizer.fit_transform(train_x)
    test_X = vectorizer.transform(test_x)
    current_time = print_timediff('vectorizering', current_time)

    x = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))

    if conf('hyper-params.on'):
        print('Hyperparams search started')
        clf = RandomForestClassifier()
        grid = GridSearchCV(
            clf,
            param_grid=conf('hyper-params.param-grid'),
            cv=conf('hyper-params.cv'),
            scoring=conf('hyper-params.scoring'),
            verbose=conf('hyper-params.verbose'),
            n_jobs=conf('hyper-params.jobs')
        )
        model_grid = grid.fit(train_X, train_y)
        print(f'Best hyperparameters are: {model_grid.best_params_}')
        print(f'Best score is: {model_grid.best_score_}')
    else:
        print('Usage of config params for RandomForestClassifier')
        clf = RandomForestClassifier(
            n_estimators=conf('cls.estimators'),
            max_depth=conf('cls.max-depth'),
        )

    clf.fit(train_X, train_y)
    prediction = clf.predict(test_X)
    current_time = print_timediff('class', current_time)

    print(f'Accuracy: {accuracy_score(test_y, prediction)}')
    print(f'F1: {f1_score(test_y, prediction, average="macro")}')
    _ = print_timediff('metrics', current_time)


if __name__ == '__main__':
    run()
