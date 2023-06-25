from src.hw_003_vector.reader.csv_reader import CsvReaderH


class DatasetReader:
    PROP_READER_ORIGINAL = 'reader.original'
    PROP_READER_PREPARED = 'reader.prepared'
    PROP_PATH_ORIGINAL = 'path.original'
    PROP_PATH_PREPARED = 'path.prepared'

    def __init__(self, **kwargs):
        reader = CsvReaderH()
        self._reader_o = kwargs.get(self.PROP_READER_ORIGINAL) if self.PROP_READER_ORIGINAL in kwargs else reader
        self._reader_p = kwargs.get(self.PROP_READER_PREPARED) if self.PROP_READER_PREPARED in kwargs else reader

    def __call__(self, *args, **kwargs):
        # PROP_PATH_ORIGINAL = 'path.original' !!!
        # PROP_PATH_PREPARED = 'path.prepared' !!!

        # 0) if PROP_PATH_ORIGINAL & PROP_PATH_PREPARED
        # 1) if PROP_PATH_ORIGINAL
        # 2) if PROP_PATH_PREPARED
        # 3) if None

        pass