import unittest

from src.hw_001_data_parsing.loader.dump_loader import SuitableFileNameGetter, FilesLoader, DumpLoader


class TestCase(unittest.TestCase):
    def test_suitable_file_name_getter(self):
        folder_path = 'test_files'
        prefix = 'tf'

        result = SuitableFileNameGetter().execute(folder_path, prefix)
        self.assertEqual(['tf0.txt', 'tf1.txt'], result)

    def test_files_loader(self):
        folder_path = 'test_files'
        files = ['tf0.txt', 'tf1.txt']
        expected = {'key0': 'value0', 'key1': 'value1'}

        result = FilesLoader().execute(folder_path, files)
        self.assertEqual(expected, result)

    def test_dump_loader(self):
        folder_path = 'test_files'
        prefix = 'tf'
        files = ['tf0.txt', 'tf1.txt']
        expected = {'key0': 'value0', 'key1': 'value1'}

        class TestGetter:
            def execute(self, folder_path: str, prefix: str) -> list:
                self.fp = folder_path
                self.p = prefix
                return files

        class TestLoader:
            def execute(self, folder_path: str, files: list) -> dict:
                self.fp = folder_path
                self.fs = files
                return expected

        getter = TestGetter()
        loader = TestLoader()
        result = DumpLoader(folder_path, prefix, file_getter=getter, loader=loader).load()
        self.assertEqual(folder_path, getter.fp)
        self.assertEqual(prefix, getter.p)
        self.assertEqual(folder_path, loader.fp)
        self.assertEqual(files, loader.fs)
        self.assertEqual(expected, result)


if __name__ == '__main__':
    unittest.main()
