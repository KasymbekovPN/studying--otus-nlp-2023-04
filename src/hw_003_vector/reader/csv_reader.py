import csv


class CsvReaderH:
    PROP_NEW_LINE = 'newLine'
    PROP_ENCODING = 'encoding'

    def __init__(self, **kwargs):
        self._new_line = kwargs.get(self.PROP_NEW_LINE) if self.PROP_NEW_LINE in kwargs else ''
        self._encoding = kwargs.get(self.PROP_ENCODING) if self.PROP_ENCODING in kwargs else 'utf-8'

    def __call__(self, *args) -> list:
        if len(args) == 0 or not isinstance(args[0], str):
            raise Exception('Path<str> is absence')

        with open(args[0], newline=self._new_line, encoding=self._encoding) as file:
            reader = csv.DictReader(file)
            result = [item for item in reader]
        return result
