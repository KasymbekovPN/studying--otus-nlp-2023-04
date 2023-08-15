import os
import shutil
import wget
import zipfile

import pandas as pd


class DefaultCheckStrategy:

    def __call__(self, *args, **kwargs) -> bool:
        directory_path = kwargs.get('directory_path')
        file_names = kwargs.get('file_names')

        file_paths = [directory_path + '/' + name for name in file_names]

        exist = True
        for file_name in file_paths:
            if not os.path.exists(file_name):
                exist = False
                break

        if not exist and os.path.isdir(directory_path):
            for file_name in os.listdir(directory_path):
                file_path = os.path.join(directory_path, file_name)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(e)

        return exist


class DefaultDownloadStrategy:
    def __init__(self, url: str, file_name='arxiv.zip') -> None:
        self._url = url
        self._file_name = file_name

    def __call__(self, *args, **kwargs) -> None:
        directory_path = kwargs.get('directory_path')
        file_path = os.path.join(directory_path, self._file_name)
        wget.download(self._url, file_path)
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(directory_path)


class DefaultPreparationStrategy:
    def __call__(self, *args, **kwargs) -> dict:
        directory_path = kwargs.get('directory_path')
        paths = kwargs.get('paths')

        result = {}
        for key, value in paths.items():
            result[key] = {
                'old': os.path.join(directory_path, value),
                'new': os.path.join(directory_path, key)
            }

        new_paths = set()
        for key, value in result.items():
            os.rename(value['old'], value['new'])
            new_paths.add(value['new'])

        for file_name in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file_name)
            try:
                if file_path not in new_paths:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
            except Exception as e:
                print(e)

        return result


class Downloader:
    def __init__(self,
                 directory_path: str,
                 paths: dict,
                 check_strategy,
                 download_strategy,
                 preparation_strategy) -> None:
        file_names = paths.keys()
        if check_strategy(directory_path=directory_path, file_names=file_names):
            data = {name: os.path.join(directory_path, name) for name in file_names}
        else:
            download_strategy(directory_path=directory_path)
            data = preparation_strategy(directory_path=directory_path, paths=paths)

        self._data = {key: pd.read_json(value, lines=True) for key, value in data.items()}

    def get(self, key: str):
        return self._data[key] if key in self._data else None
