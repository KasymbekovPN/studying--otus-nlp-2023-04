import unittest
import re

from parameterized import parameterized
from src.hw_003_vector.preparation.regexp_preparators import RegexPreparator, HtmlTagsPreparator, PunctuationPreparator, SpacePreparator


class TestCase(unittest.TestCase):

    @parameterized.expand([
        [{'review': 'aaa'}, {'review': 'bbb'}],
        [{'review': 'ccc'}, {'review': 'ccc'}],
        [{'review': 'abc'}, {'review': 'bbc'}]
    ])
    def test_reg_filter(self, original_datum: dict, expected_datum: dict):
        f = RegexPreparator(re.compile(r'a'), 'b')
        result_datum = f(datum=original_datum)
        self.assertEqual(expected_datum, result_datum)

    @parameterized.expand([
        [{'review': 'simple text'}, {'review': 'simple text'}],
        [{'review': '10 > 5'}, {'review': '10 > 5'}],
        [{'review': '10 < 100 but 100 > 50'}, {'review': '10 < 100 but 100 > 50'}],
        [{'review': 'text before <p class="a-b-c">value<p /> text after'}, {'review': 'text before  value  text after'}],
        [{'review': '<br /><br />The first thing'}, {'review': '  The first thing'}]
    ])
    def test_tags_filter(self, original_datum: dict, expected_datum: dict):
        f = HtmlTagsPreparator()
        result_datum = f(datum=original_datum)
        self.assertEqual(expected_datum, result_datum)

    @parameterized.expand([
        [{'review': 'Hello'}, {'review': 'Hello'}],
        [{'review': 'Hello, world!!!'}, {'review': 'Hello  world '}],
        [{'review': 'a_*.,!?()_*b'}, {'review': 'a b'}]
    ])
    def test_punctuation_filter(self, original_datum: dict, expected_datum: dict):
        f = PunctuationPreparator()
        result_datum = f(datum=original_datum)
        self.assertEqual(expected_datum, result_datum)

    @parameterized.expand([
        [{'review': '   '}, {'review': ' '}],
        [{'review': 'hello     world'}, {'review': 'hello world'}],
        [{'review': 'before \f\n\r\t\v after'}, {'review': 'before after'}]
    ])
    def test_space_normalization(self, original_datum: dict, expected_datum: dict):
        f = SpacePreparator()
        result_datum = f(datum=original_datum)
        self.assertEqual(expected_datum, result_datum)


if __name__ == '__main__':
    unittest.main()
