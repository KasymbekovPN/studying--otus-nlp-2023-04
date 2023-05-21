import unittest

from os.path import join
from src.hw_001_data_parsing.loader.dump_loader import DumpSuitableFileNamesComputer, DatasetSuitableFileNamesComputer,\
    DumpFilesReader, DatasetFilesReader, DumpReader


class TestCase(unittest.TestCase):

    def test_dump_suitable_name_computer(self):
        folder_path = 'test_files'
        prefix = 'tf'

        result = DumpSuitableFileNamesComputer(folder_path, prefix)()
        self.assertEqual(['tf0.txt', 'tf1.txt'], result)

    def test_dataset_suitable_name_computer(self):
        folder_path = 'test_files'
        name = 'dataset.csv'

        result = DatasetSuitableFileNamesComputer(folder_path, name)()
        self.assertEqual([join(folder_path, name)], result)

    def test_dump_file_reader(self):
        folder_path = 'test_files'
        files = ['tf0.txt', 'tf1.txt']
        expected = {'key0': 'value0', 'key1': 'value1'}

        result = DumpFilesReader(folder_path)(files=files)
        self.assertEqual(expected, result)

    def test_dataset_file_reader(self):
        folder_path = 'test_files'
        file_name = 'dataset.csv'
        files = [file_name]
        expected_data = {'id0': {'view_counter': 'view_counter0', 'datetime': 'datetime0', 'article': 'article0'},
                         'id1': {'view_counter': 'view_counter1', 'datetime': 'datetime1', 'article': 'article1'}}

        result = DatasetFilesReader(folder_path)(files=files)
        self.assertEqual(expected_data, result[file_name])

    def test_dump_reader(self):
        quantity = 10
        files = [f'file_path{i}' for i in range(0, quantity)]
        expected = {f: f for f in files}

        class TestNamesComputer:
            def __init__(self, fs):
                self._files = fs

            def __call__(self, *args, **kwargs) -> list:
                return self._files

        class TestReader:
            def __call__(self, *args, **kwargs):
                return {f: f for f in kwargs.get('files')}

        result = DumpReader(TestNamesComputer(files), TestReader())()
        self.assertEqual(expected, result)


if __name__ == '__main__':
    unittest.main()
