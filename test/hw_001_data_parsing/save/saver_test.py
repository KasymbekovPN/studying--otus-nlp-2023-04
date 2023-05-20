from parameterized import parameterized
import unittest
import os

from os import listdir
from os.path import join, isfile
from src.hw_001_data_parsing.save.saver import Rewriter, Eraser, Saver


class TestCase(unittest.TestCase):

    @parameterized.expand([
        ('some content 0', False),
        ('some content 1', False),
        ('some content 2', True)
    ])
    def test_rewriter(self, content: str, erase_file: bool):
        file_name = 'file_name.txt'
        Rewriter()(path=file_name, content=content)

        try:
            with open(file_name, 'r', encoding='utf-8') as file:
                taken_content = file.read()
        except:
            taken_content = None

        self.assertEqual(content, taken_content)
        if erase_file:
            os.remove(file_name)

    def test_eraser(self):
        folder_path = './test_files'
        file_name_prefix = 'prefix_'
        quantity = 3

        for i in range(0, quantity):
            with open(f'{folder_path}/{file_name_prefix}{i}.txt', 'w', encoding='utf-8') as f:
                f.write('')

        Eraser()(folder_path=folder_path, prefix=file_name_prefix)

        files = [f for f in listdir(folder_path) if isfile(join(folder_path, f)) and f[:len(file_name_prefix)] == file_name_prefix]
        self.assertEqual(0, len(files))

    def test_saver(self):
        class TestEraser:
            def __call__(self, *args, **kwargs):
                self.called = True

        class TestWriter:
            def __call__(self, *args, **kwargs):
                self.path = kwargs.get('path')
                self.content = kwargs.get('content')

        class TestHolder:
            def __init__(self, path: str, content: str):
                self._data = [(path, content)]
                self.folder_path = ''
                self.prefix = ''

            def next(self):
                if len(self._data) == 0:
                    return None
                return self._data.pop(0)

        expected_path = 'some.path'
        expected_content = 'some.content'
        eraser = TestEraser()
        writer = TestWriter()
        Saver(writer, eraser).save(TestHolder(expected_path, expected_content))

        self.assertEqual(True, eraser.called)
        self.assertEqual(expected_path, writer.path)
        self.assertEqual(expected_content, writer.content)


if __name__ == '__main__':
    unittest.main()
