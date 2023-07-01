import copy
import csv

from src.hw_003_vector.configurator.configurator import Configurator
from src.hw_003_vector.reader.dataset_reader import DatasetReader
from src.hw_003_vector.preparation.preparation import Preparation
from src.hw_003_vector.preparation.case_preparator import CasePreparator
from src.hw_003_vector.preparation.regexp_preparators import PunctuationPreparator, SpacePreparator, HtmlTagsPreparator
from src.hw_003_vector.preparation.text_lemm_preparator import TextLemmPreparator
from src.hw_003_vector.preparation.sentiment_read_preparator import SentimentReadPreparator
from src.hw_003_vector.preparation.sentiment_write_preparator import SentimentWritePreparator


def write_prepared_dataset(dataset: list, conf: Configurator):
    with open(conf('path.dataset.prepared'), 'w', newline='', encoding='utf-8') as file:
        fieldnames = ['review', 'sentiment']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        [writer.writerow(datum) for datum in dataset]


def get_dataset(conf: Configurator) -> list[dict] | None:
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
        print(6)
        prepared = copy.deepcopy(result)
        write_preparator = Preparation(SentimentWritePreparator())
        prepared = [write_preparator(datum) for datum in prepared]
        saver(prepared, conf)

    return result


def run():
    conf = Configurator()

    dataset = get_dataset(conf)
    if dataset is None:
        return


if __name__ == '__main__':
    run()
