import unittest
from parameterized import parameterized

from src.hw_001_data_parsing.eda.eda_executor import ViewCounterConverter, MostViewedArticlesTask, \
    MostFrequentWordsTask, EDAExecutor


class TestCase(unittest.TestCase):

    @parameterized.expand([
        ('-1', 0),
        ('1', 1),
        ('12', 12),
        ('123', 123),
        ('1K', 1_000),
        ('1.2K', 1_200),
        ('1.2.3K', 0),
        ('2M', 2_000_000),
        ('2.3M', 2_300_000),
        ('2.3.4.M', 0),
        ('2.3Z', 0),
        ('K3.4', 0)
    ])
    def test_view_counter_converter(self, text: str, expected: int):
        converter = ViewCounterConverter()
        self.assertEqual(expected, converter(text=text))

    def test_most_viewed_articles_task(self):
        top_size = 6
        base = 100
        quantity = 20
        dataset = {f'key{i}': {'view_counter': f'{i//2}'} for i in range(base, base + quantity)}
        task = MostViewedArticlesTask(top_size)
        task(dataset=dataset)

        expected = {f'key{i}': i//2 for i in reversed(range(base + quantity - top_size, base + quantity))}
        self.assertEqual(expected, task.result[task.KEY])

    def test_most_frequent_words_task(self):
        top_size = 3
        words = [f'word{i}' for i in range(0, 20)]
        dataset = {
            'key0': {'article': f'{words[0]} {words[1]} {words[2]} {words[3]} {words[3]} {words[4]}'},
            'key1': {'article': f'{words[1]} {words[5]} {words[5]} {words[6]} {words[7]} {words[8]}'},
            'key2': {'article': f'{words[9]} {words[9]} {words[9]} {words[10]} {words[10]} {words[10]}'},
            'key3': {'article': f'{words[11]} {words[12]} {words[12]} {words[12]} {words[12]} {words[12]}'},
            'key4': {'article': f'{words[13]} {words[13]} {words[13]} {words[13]} {words[13]} {words[13]}'},
            'key5': {'article': f'{words[14]} {words[15]} {words[16]} {words[16]} {words[17]} {words[18]}'},
            'key6': {'article': f'{words[1]} {words[2]} {words[2]} {words[3]} {words[3]} {words[19]}'}
        }
        excluded = [words[8], words[13]]
        expected = {
            words[12]: 5,
            words[3]: 4,
            words[1]: 3,
            # words[2]: 3,
            # words[9]: 3,
            # words[10]: 3,
            # words[16]: 2,
            # words[0]: 1,
            # words[4]: 1,
            # words[5]: 2,
            # words[6]: 1,
            # words[7]: 1,
            # words[11]: 1,
            # words[14]: 1,
            # words[15]: 1,
            # words[17]: 1,
            # words[18]: 1,
            # words[19]: 1
        }

        task = MostFrequentWordsTask(top_size, excluded)
        task(dataset=dataset)

        self.assertEqual(expected, task.result[task.KEY])

    def test_eda_executor(self):
        class TestTask:
            def __call__(self, *args, **kwargs):
                self.dataset = kwargs.get('dataset')

        test_quantity = 10
        tasks = [TestTask() for i in range(0, test_quantity)]
        dataset = {'key': 'value'}

        executor = EDAExecutor(dataset)
        for task in tasks:
            executor.add_task(task)
        executor.start()

        for task in tasks:
            self.assertEqual(dataset, task.dataset)


if __name__ == '__main__':
    unittest.main()
