import unittest

from parameterized import parameterized
from src.hw_001_data_parsing.save.content_holder import HashNameComputer, PageContentHolder, DatasetContentHolder


class TestCase(unittest.TestCase):

    @parameterized.expand([
        ('file_prefix_', 'value0,', 'file_prefix_30fcba6e26b707ca91ac87dbdcb42283.htm'),
        ('file_prefix_', 'value1,', 'file_prefix_06d6ef84b4ff78a0cddf1eabc5e2eeaa.htm'),
        ('file_prefix_', 'value2,', 'file_prefix_38b07bfb951740b38f770f505bc9a1a3.htm')
    ])
    def test_hash_name_computer(self, prefix, value, expected):
        computer = HashNameComputer()
        name = computer(prefix=prefix, value=value)
        self.assertEqual(expected, name)

    def test_page_content_holder(self):
        expected_name = 'some.name'
        expected_prefix = 'some.prefix'
        expected_folder_path = 'some.folder.path'

        class TestHashNameComputer:
            def __call__(self, *args, **kwargs) -> str:
                return expected_name

        quantity = 3
        source = {f'key{i}': f'content{i}' for i in range(0, quantity)}
        content_holder = PageContentHolder(source, expected_folder_path, expected_prefix, TestHashNameComputer())

        for idx in range(0, quantity):
            result = content_holder.next()
            expected_path = f'{expected_folder_path}/{expected_name}'
            self.assertEqual(expected_path, result[0])
            expected_content = f'key{idx}\ncontent{idx}'
            self.assertEqual(expected_content, result[1])
        self.assertEqual(None, content_holder.next())

    def test_dataset_content_holder(self):
        quantity = 2
        data = {}
        for i in range(0, quantity):
            data[f'https://link{i}.com'] = {
                'view_counter': f'{i}',
                'datetime': f'some.datetime{i}',
                'article': f'article {i}'
            }

        expected_folder_path = 'some.folder.path'
        expected_path = f'{expected_folder_path}/dataset.csv'

        expected_content = 'id,view_counter,datetime,article\n'
        for k, v in data.items():
            expected_content += f'{k},{v.get("view_counter")},{v.get("datetime")},{v.get("article")}\n'

        content_holder = DatasetContentHolder(data, expected_folder_path)
        result = content_holder.next()
        self.assertEqual(expected_path, result[0])
        self.assertEqual(expected_content, result[1])
        self.assertEqual(None, content_holder.next())


if __name__ == '__main__':
    unittest.main()
