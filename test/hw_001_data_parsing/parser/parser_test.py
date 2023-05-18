import unittest
from bs4 import BeautifulSoup

from src.hw_001_data_parsing.parser.parser import Parser, CollectPageLinksTask, CollectPublishedDatetime, CollectViewCounter


class TestCase(unittest.TestCase):

    def test_parse_in_parser(self):
        class TestTask:
            def __init__(self):
                self._key = None
                self._soup = None

            def execute(self, key: str, soup: 'BeautifulSoup'):
                self._soup = soup
                self._key = key

        page = '<div></div'
        expected_soup = BeautifulSoup(page, 'html.parser')
        expected_key = 'some_key'

        task = TestTask()
        Parser().add_task(task).parse(expected_key, page)

        self.assertEqual(expected_key, task._key)
        self.assertEqual(expected_soup, task._soup)

    def test_parse_dict_in_parser(self):
        class TestTask:
            def __init__(self):
                self._keys = []
                self._soups = []

            def execute(self, key: str, soup: 'BeautifulSoup'):
                self._soups.append(soup)
                self._keys.append(key)

        amount = 10
        d = {f'key{i}': f'<div>{i}</div>' for i in range(0, amount)}
        expected_keys = list(d.keys())
        expected_soups = [BeautifulSoup(v, 'html.parser') for v in d.values()]

        task = TestTask()
        Parser().add_task(task).parse_dict(d)

        self.assertEqual(expected_keys, task._keys)
        self.assertEqual(expected_soups, task._soups)

    def test_collect_page_link_task(self):
        expected_links = ["/ru/articles/734928/",
                          "/ru/articles/734929/",
                          "/ru/articles/734930/",
                          "/ru/articles/734932/"]
        page0 = """
        <div>
            <a
                href="/ru/articles/734928/"
                data-test-id="article-snippet-title-link"
                data-article-link="true"
                class="tm-title__link"
            >
                <span>Поднимаем Kubernetes с нуля</span>
            </a>
            <a
                href="/ru/articles/734929/"
                data-test-id="article-snippet-title-link"
                data-article-link="true"
                class="tm-title__link"
            >
                <span>Поднимаем Kubernetes с нуля</span>
            </a>
            <a
                href="/ru/articles/734930/"
                data-test-id="article-snippet-title-link"
                data-article-link="true"
                class="tm-title__link"
            >
                <span>Поднимаем Kubernetes с нуля</span>
            </a>
            <a
                href="/ru/articles/734931/"
                data-test-id="article-snippet-title-link"
                data-article-link="true"
                class="wrong-class"
            >
                <span>Поднимаем Kubernetes с нуля</span>
            </a>
        </div>
        """

        page1 = """
        <div>
            <a
                href="/ru/articles/734932/"
                data-test-id="article-snippet-title-link"
                data-article-link="true"
                class="tm-title__link"
            >
                <span>Поднимаем Kubernetes с нуля</span>
            </a>
        </div>
        """

        task = CollectPageLinksTask('a', 'tm-title__link')\
            .execute('', BeautifulSoup(page0, 'html.parser'))\
            .execute('', BeautifulSoup(page1, 'html.parser'))

        self.assertEqual(expected_links, task.attrs[task.KEY])

    def test_collect_datetime_task(self):
        expected_datetime = '2023-01-23T11:31:09.000Z'
        expected_key = 'some.key'
        class_ = 'tm-article-datetime-published'

        page = f"""
        <div>
            <span class="tm-article-datetime-published">
                <time datetime="{expected_datetime}" title="2023-01-23, 14:31">
                    23  янв   в 14:31
                </time>
            </span>
        </div>"""
        soup = BeautifulSoup(page, 'html.parser')

        task = CollectPublishedDatetime('span', class_)
        task.execute(expected_key, soup)

        self.assertEqual(True, expected_key in task.attrs)
        self.assertEqual(expected_datetime, task.attrs[expected_key])

    def test_collect_view_counter_task(self):
        class_ = 'tm-icon-counter__value'
        expected_count = '2.1K'
        expected_key = 'some.key'

        page = f'<div><span class="tm-icon-counter__value">{expected_count}</span></div>'
        soup = BeautifulSoup(page, 'html.parser')

        task = CollectViewCounter('span', class_)
        task.execute(expected_key, soup)

        self.assertEqual(True, expected_key in task.attrs)
        self.assertEqual(expected_count, task.attrs[expected_key])


if __name__ == '__main__':
    unittest.main()
