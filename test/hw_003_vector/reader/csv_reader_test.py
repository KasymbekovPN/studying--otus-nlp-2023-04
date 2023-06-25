import unittest

from src.hw_003_vector.reader.csv_reader import read_csv_with_header


class TestCase(unittest.TestCase):
    def test_csv_reading_with_header(self):
        expected = [
            {'header0': 'value00', 'header1': 'value10', 'header2': 'value20'},
            {'header0': 'value01', 'header1': 'value11', 'header2': 'value21'},
            {'header0': 'value02', 'header1': 'value12', 'header2': 'value22'}
        ]

        result = read_csv_with_header('./csv_with_header.csv')
        self.assertEqual(expected, result)


if __name__ == '__main__':
    unittest.main()
