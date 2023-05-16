import unittest
from bs4 import BeautifulSoup

from src.hw_001_data_parsing.parser.parser import CollectPageLinksTask


class TestCase(unittest.TestCase):

    def test_collect_page_link_task(self):
        expected_links = ["/ru/articles/734928/",
                          "/ru/articles/734929/",
                          "/ru/articles/734930/"]
        page = """
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

        key = 'links'
        soup = BeautifulSoup(page, 'html.parser')
        task = CollectPageLinksTask('a', 'tm-title__link').execute(soup)

        self.assertEqual(expected_links, task.attrs[key])


if __name__ == '__main__':
    unittest.main()
