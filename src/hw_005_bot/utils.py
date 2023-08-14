# todo ??? make class with strategies

import os
import wget
import zipfile


RUSSE_FILE_NAMES = (
    'test.jsonl',
    'train.jsonl',
    'val.jsonl'
)


def download_dataset(url: str, file_path: str):
    if os.path.exists(file_path):
        print(f'Raw dataset is already downloaded | "{file_path}".')
    else:
        wget.download(url, file_path)
        print(f'\nDataset is downloaded | "{file_path}".')


def extract_archive(arch_path: str, output_path: str):
    with zipfile.ZipFile(arch_path, 'r') as zip_ref:
        zip_ref.extractall(output_path)
    print(f'It is extracted to "{output_path}"')


def prepare_russe_dataset_directory(directory_path: str):
    directory_with_files = directory_path + '/RUSSE'
    paths = [(directory_with_files + '/' + name, directory_path + '/' + name, ) for name in RUSSE_FILE_NAMES]

    for path_pair in paths:
        os.rename(path_pair[0], path_pair[1])


# todo del
# if __name__ == '__main__':
#     directory_path = './directory/path'
#     prepare_russe_dataset_directory(directory_path)
#     pass
