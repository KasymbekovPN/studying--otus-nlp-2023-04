import unittest

from src.hw_001_data_parsing.datasource.links_source import PerforatingLinkCreator, FeedLinksCreator


class TestCase(unittest.TestCase):
    def test_perforating_link_creation(self):
        expected_links = [f'https://link{i}.org' for i in range(0, 10)]
        creator = PerforatingLinkCreator(expected_links)
        self.assertEqual(expected_links, creator())

    def test_feed_page_link_creation(self):
        link_quantity = 10
        expected_links = [f'https://habr.com/ru/all/page{i+1}/' for i in range(0, link_quantity)]
        creator = FeedLinksCreator(link_quantity)
        self.assertEqual(expected_links, creator())


if __name__ == '__main__':
    unittest.main()
