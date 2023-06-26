from src.hw_003_vector.reader.csv_reader import CsvReaderH


class DatasetReader:
    PROP_READER_ORIGINAL = 'reader_original'
    PROP_READER_PREPARED = 'reader_prepared'
    PROP_PATH_ORIGINAL = 'path_original'
    PROP_PATH_PREPARED = 'path_prepared'

    def __init__(self, **kwargs) -> None:
        reader = CsvReaderH()
        self._reader_o = kwargs.get(self.PROP_READER_ORIGINAL) if self.PROP_READER_ORIGINAL in kwargs else reader
        self._reader_p = kwargs.get(self.PROP_READER_PREPARED) if self.PROP_READER_PREPARED in kwargs else reader
        self._original = None
        self._prepared = None

    def __call__(self, *args, **kwargs) -> None:
        original_path = kwargs.get(self.PROP_PATH_ORIGINAL) if self.PROP_PATH_ORIGINAL in kwargs else None
        prepared_path = kwargs.get(self.PROP_PATH_PREPARED) if self.PROP_PATH_PREPARED in kwargs else None

        if original_path is not None:
            self._original = self._reader_o(original_path)
        if prepared_path is not None:
            self._prepared = self._reader_p(prepared_path)

    @property
    def original(self) -> list | None:
        return self._original

    @original.setter
    def original(self, value) -> None:
        raise Exception('[original] unsupported setting')

    @property
    def prepared(self) -> list | None:
        return self._prepared

    @prepared.setter
    def prepared(self, value) -> None:
        raise Exception('[prepared] unsupported setting')
