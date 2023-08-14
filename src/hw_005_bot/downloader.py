import os
import wget
import zipfile


class DefaultCheckStrategy:
    def __call__(self, *args, **kwargs) -> bool:
        directory_path = kwargs.get('directory_path')
        file_names = kwargs.get('file_names')

        file_paths = [directory_path + '/' + name for name in file_names]

        # check file_paths existence

        # if true return true
        # else clear directory and return false

        pass


class DefaultDownloadStrategy:
    def __call__(self, *args, **kwargs):
        pass

# todo ???
# def download_dataset(url: str, file_path: str):
#     if os.path.exists(file_path):
#         print(f'Raw dataset is already downloaded | "{file_path}".')
#     else:
#         wget.download(url, file_path)
#         print(f'\nDataset is downloaded | "{file_path}".')


class Downloader:
    def __init__(self,
                 directory_path: str,
                 file_names: tuple[str]) -> None:
        pass
