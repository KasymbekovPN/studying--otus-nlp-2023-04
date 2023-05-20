import csv

from os.path import isfile, join
from os import listdir


class DumpSuitableFileNamesComputer:
    def __init__(self, folder_path: str, prefix: str) -> None:
        self._folder_path = folder_path
        self._prefix = prefix

    def __call__(self, *args, **kwargs) -> list:
        return [f for f in listdir(self._folder_path) if
                isfile(join(self._folder_path, f)) and f[:len(self._prefix)] == self._prefix]


class DatasetSuitableFileNamesComputer:
    def __init__(self, folder_path: str, name='dataset.csv') -> None:
        self._data = [join(folder_path, name)]

    def __call__(self, *args, **kwargs) -> list:
        return self._data


class DumpFilesReader:
    def __init__(self, folder_path) -> None:
        self._folder_path = folder_path

    def __call__(self, *args, **kwargs) -> dict:
        files = kwargs.get('files')
        result = {}
        for file in files:
            path = join(self._folder_path, file)
            with open(path, 'r', encoding='utf-8') as f:
                key = f.readline().strip()
                result[key] = f.read()

        return result


class DatasetFilesReader:
    def __init__(self, folder_path: str) -> None:
        self._folder_path = folder_path

    def __call__(self, *args, **kwargs) -> dict:
        files = kwargs.get('files')
        result = {}
        for file in files:
            path = join(self._folder_path, file)
            with open(path, 'r', encoding='utf-8') as f:
                csv_reader = csv.reader(f)
                next(csv_reader)
                result_item = {}
                for item in csv_reader:
                    result_item[item[0]] = {'view_counter': item[1], 'datetime': item[2], 'article': item[3]}
                result[file] = result_item
        return result


class DumpReader:
    def __init__(self,
                 names_computer,
                 reader) -> None:
        self._names_computer = names_computer
        self._reader = reader

    def __call__(self, *args, **kwargs):
        files = self._names_computer()
        return self._reader(files=files)
