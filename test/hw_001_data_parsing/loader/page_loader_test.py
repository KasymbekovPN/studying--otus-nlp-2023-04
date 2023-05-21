from parameterized import parameterized
import unittest

from src.hw_001_data_parsing.loader.page_loader import get_page_raw_data, run_load_task


class TestCase(unittest.TestCase):

    @parameterized.expand([
        (200, 'https://path0.com', 'https://path0.com'),
        (200, 'https://path1.com', 'https://path1.com'),
        (400, 'https://path2.com', ''),
    ])
    def test_raw_page_data_getting(self, t, u, raw_text):
        class TestResponse:
            def __init__(self, status_code: int, text: str) -> None:
                self.status_code = status_code
                self.text = text

        def get_test_page(url: str, timeout: int) -> 'response':
            return TestResponse(timeout, url)

        result = get_page_raw_data(u, t, getter=get_test_page)

        self.assertEqual(True, isinstance(result, dict))
        self.assertEqual({'status_code': t, 'raw_text': raw_text, 'url': u}, result)

    def test_run_load_task_if_link_generator_absence(self):
        result = run_load_task()
        self.assertEqual(0, len(result))

    def test_run_load_task(self):
        def get_test_page_raw_data(url: str, timeout: int, getter=None):
            return {'status_code': 200, 'raw_text': ('text: ' + url), 'url': url}

        amount = 4
        timeout = 3
        link_pack = ['https://path' + str(i) + '.com' for i in range(0, amount)]

        result = run_load_task(getter=get_test_page_raw_data, link_pack=link_pack, timeout=timeout)

        expected = {'https://path' + str(i) + '.com': 'text: https://path' + str(i) + '.com' for i in range(0, amount)}
        self.assertEqual(True, isinstance(result, dict))
        self.assertEqual(expected, result)


if __name__ == '__main__':
    unittest.main()
