from os.path import isfile, join
from os import listdir


class SuitableFileNameGetter:
    def execute(self, folder_path: str, prefix: str) -> list:
        return [f for f in listdir(folder_path) if isfile(join(folder_path, f)) and f[:len(prefix)] == prefix]


class FilesLoader:
    def execute(self, folder_path: str, files: list) -> dict:
        result = {}
        for file in files:
            path = join(folder_path, file)
            with open(path, 'r', encoding='utf-8') as f:
                key = f.readline().strip()
                result[key] = f.read()

        return result


class DumpLoader:
    def __init__(self,
                 folder_path: str,
                 file_prefix: str,
                 file_getter=SuitableFileNameGetter(),
                 loader=FilesLoader()) -> None:
        self._folder_path = folder_path
        self._file_prefix = file_prefix
        self._file_getter = file_getter
        self._loader = loader

    def load(self) -> dict:
        files = self._file_getter.execute(self._folder_path, self._file_prefix)
        return self._loader.execute(self._folder_path, files)
