import unittest
from bs4 import BeautifulSoup

from src.hw_001_data_parsing.parser.parser import Parser, CollectPageLinksTask, CollectPublishedDatetimeTask, \
    CollectViewCounterTask, CollectArticleTask


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
        expected_links = ["https://habr.com/ru/articles/734928/",
                          "https://habr.com/ru/articles/734929/",
                          "https://habr.com/ru/articles/734930/",
                          "https://habr.com/ru/articles/734932/"]
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

        task = CollectPageLinksTask('a', 'tm-title__link') \
            .execute('', BeautifulSoup(page0, 'html.parser')) \
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

        task = CollectPublishedDatetimeTask('span', class_)
        task.execute(expected_key, soup)

        self.assertEqual(True, expected_key in task.attrs)
        self.assertEqual(expected_datetime, task.attrs[expected_key])

    def test_collect_view_counter_task(self):
        class_ = 'tm-icon-counter__value'
        expected_count = '2.1K'
        expected_key = 'some.key'

        page = f'<div><span class="tm-icon-counter__value">{expected_count}</span></div>'
        soup = BeautifulSoup(page, 'html.parser')

        task = CollectViewCounterTask('span', class_)
        task.execute(expected_key, soup)

        self.assertEqual(True, expected_key in task.attrs)
        self.assertEqual(expected_count, task.attrs[expected_key])

    def test_collect_article_task(self):
        expected_key = 'some.key'
        words = [f'Word{w}' for w in range(0, 14)]
        expected_words = [w.lower() for w in words]
        page = f"""
        <div>
            <div class="article-formatted-body article-formatted-body article-formatted-body_version-2">
                <p>{words[0]}</p>
                <p>{words[1]}</p>
                <a href="https://habr.com/ru/users/NewTechAudit/posts/" rel="noopener noreferrer nofollow">{words[2]}</a>
                <figure class="full-width "><img src="https://habrastorage.org/getpro/habr/upload_files/f5a/0bc/288/f5a0bc2880d7bfb2889ca5a9ade49dc5.jpg" width="1280" height="720"><figcaption></figcaption></figure>
                <pre>
                    <code>
                        {{{words[3]}: {words[4]}, {words[5]}: ‘{words[6]}’, {words[7]}: ‘{words[8]}’, {words[9]}: ‘{words[10]}’}}
                    </code>
                </pre>
                <ul>
                    <li><p>{words[11]}</p></li>
                    <li><p>{words[12]}</p></li>
                    <li><p>{words[13]}</p></li>
                </ul>
            </div>
        </div>
                """

        soup = BeautifulSoup(page, 'html.parser')
        task = CollectArticleTask('div', 'article-formatted-body')
        task.execute(expected_key, soup)

        expected_text = ''
        delimiter = ''
        for w in expected_words:
            expected_text += delimiter + w
            delimiter = ' '
        self.assertEqual(True, expected_key in task.attrs)
        self.assertEqual(expected_text, task.attrs[expected_key])


if __name__ == '__main__':
    unittest.main()
