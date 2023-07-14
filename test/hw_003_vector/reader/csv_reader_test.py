import unittest

from src.hw_003_vector.reader.csv_reader import read_csv_with_header, CsvReaderH


class TestCase(unittest.TestCase):

    def test_csv_reader_h_if_zero_arg_absence(self):
        with self.assertRaises(Exception) as context:
            CsvReaderH()()
        self.assertTrue('Path<str> is absence' in context.exception.args)

    def test_csv_reader_h_if_zero_arg_not_str(self):
        with self.assertRaises(Exception) as context:
            CsvReaderH()()
        self.assertTrue('Path<str> is absence' in context.exception.args)

    def test_csv_reader_h(self):
        expected = [
            {'header0': 'value00', 'header1': 'value10', 'header2': 'value20'},
            {'header0': 'value01', 'header1': 'value11', 'header2': 'value21'},
            {'header0': 'value02', 'header1': 'value12', 'header2': 'value22'}
        ]

        result = CsvReaderH()('./csv_with_header.csv')
        self.assertEqual(expected, result)


if __name__ == '__main__':
    unittest.main()
