import pathlib

from src.hw_003_vector.configurator.configurator import Configurator


def read_dataset(conf: Configurator):
    prepared_dataset_path = pathlib.Path(conf('path.dataset.prepared'))

    # 'path.dataset.original': '../../datasets/IMDB Dataset.csv',
    # : '../../datasets/prepared IMDB Dataset.csv',  # filtered & lemmatized dataset

    pass


def run():
    conf = Configurator()
    x = read_dataset(conf)


if __name__ == '__main__':
    run()
