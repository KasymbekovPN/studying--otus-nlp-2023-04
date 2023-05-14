from parameterized import parameterized
import unittest
import os
import hashlib

from src.hw_001_data_parsing.save.saver import rewrite_file, calculate_file_name


class TestCase(unittest.TestCase):

    @parameterized.expand([
        ('some content 0', False),
        ('some content 1', False),
        ('some content 2', True)
    ])
    def test_file_rewriting(self, content: str, erase_file: bool):
        file_name = 'file_name.txt'
        rewrite_file(file_name, content)

        try:
            with open(file_name, 'r') as file:
                taken_content = file.read()
        except:
            taken_content = None

        self.assertEqual(content, taken_content)
        if erase_file:
            os.remove(file_name)

    @parameterized.expand([
        ('file_prefix_', 'value0,', 'file_prefix_30fcba6e26b707ca91ac87dbdcb42283.htm'),
        ('file_prefix_', 'value1,', 'file_prefix_06d6ef84b4ff78a0cddf1eabc5e2eeaa.htm'),
        ('file_prefix_', 'value2,', 'file_prefix_38b07bfb951740b38f770f505bc9a1a3.htm')
    ])
    def test_file_name_calculation(self, prefix: str, value: str, expected: str):
        result = calculate_file_name(prefix, value)
        self.assertEqual(expected, result)


if __name__ == '__main__':
    unittest.main()
