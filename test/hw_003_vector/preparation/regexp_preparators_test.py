import unittest
import re

from parameterized import parameterized
from src.hw_003_vector.preparation.regexp_preparators import RegexPreparator, HtmlTagsPreparator, PunctuationPreparator, SpacePreparator


# todo del
class TestCase(unittest.TestCase):

    @parameterized.expand([
        ['aaa', 'bbb'],
        ['ccc', 'ccc'],
        ['abc', 'bbc']
    ])
    def test_reg_filter(self, original_text: str, expected_text: str):
        f = RegexPreparator(re.compile(r'a'), 'b')
        result_text = f(text=original_text)
        self.assertEqual(expected_text, result_text)

    @parameterized.expand([
        ['simple text', 'simple text'],
        ['10 > 5', '10 > 5'],
        ['10 < 100 but 100 > 50', '10 < 100 but 100 > 50'],
        ['text before <p class="a-b-c">value<p /> text after', 'text before  value  text after'],
        ['<br /><br />The first thing', '  The first thing']
    ])
    def test_tags_filter(self, original_text: str, expected_text):
        f = HtmlTagsPreparator()
        result_text = f(text=original_text)
        self.assertEqual(expected_text, result_text)

    @parameterized.expand([
        ['Hello', 'Hello'],
        ['Hello, world!!!', 'Hello  world '],
        ['a_*.,!?()_*b', 'a b']
    ])
    def test_punctuation_filter(self, original_text: str, expected_text: str):
        f = PunctuationPreparator()
        result_text = f(text=original_text)
        self.assertEqual(expected_text, result_text)

    @parameterized.expand([
        ['   ', ' '],
        ['hello     world', 'hello world'],
        ['before \f\n\r\t\v after', 'before after']
    ])
    def test_space_normalization(self, original_text: str, expected_text: str):
        f = SpacePreparator()
        result_text = f(text=original_text)
        self.assertEqual(expected_text, result_text)


if __name__ == '__main__':
    unittest.main()