import unittest

from src.hw_001_data_parsing.parser.collector import Collector


class TestCase(unittest.TestCase):
    def test_collector(self):
        topic0 = 'topic0'
        topic1 = 'topic1'
        part0 = {'key0': 'content0', 'key1': 'content1'}
        part1 = {'key0': 'content2', 'key1': 'content3'}
        expected = {
            'key0': {'topic0': 'content0', 'topic1': 'content2'},
            'key1': {'topic0': 'content1', 'topic1': 'content3'}
        }

        result = Collector().add(topic0, part0).add(topic1, part1).get()
        self.assertEqual(expected, result)


if __name__ == '__main__':
    unittest.main()
