import pathlib

from src.hw_003_vector.configurator.configurator import Configurator
from src.hw_003_vector.reader.dataset_reader import DatasetReader


def prepare_and_save_dataset(original: list, conf: Configurator) -> list:
    # todo !!!
    pass


def run():
    conf = Configurator()

    dataset_reader = DatasetReader()
    dataset_reader(path_original=conf('path.dataset.original'), path_prepared=conf('path.dataset.prepared'))

    # todo !!!
    if dataset_reader.original is None:
        print('No one dataset')
        return
    prepare_and_save_dataset(dataset_reader.original, conf)

    # restore
    # if dataset_reader.prepared is None:
    #     if dataset_reader.original is None:
    #         print('No one dataset')
    #         return
    #     raw_dataset = prepare_and_save_dataset(dataset_reader.original, conf)
    # else:
    #     raw_dataset = dataset_reader.prepared
    # print(raw_dataset)


if __name__ == '__main__':
    run()
