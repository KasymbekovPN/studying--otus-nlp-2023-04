from os import listdir, remove
from os.path import join, isfile

from src.hw_001_data_parsing.save.content_holder import ContentHolder


class Rewriter:
    def __init__(self, encoding='utf-8'):
        self._encoding = encoding

    def __call__(self, *args, **kwargs):
        try:
            with open(kwargs.get('path'), 'w', encoding='utf-8') as f:
                f.write(kwargs.get('content'))
            return True
        except Exception as ex:
            print(ex)
            return False


class Eraser:
    def __call__(self, *args, **kwargs):
        path = kwargs.get('folder_path')
        prefix = kwargs.get('prefix')
        files = [f for f in listdir(path) if isfile(join(path, f)) and f[:len(prefix)] == prefix]
        for f in files:
            remove(join(path, f))


class Saver:
    def __init__(self,
                 writer=Rewriter(),
                 eraser=Eraser()) -> None:
        self._writer = writer
        self._eraser = eraser

    def save(self, content_holder: 'ContentHolder'):
        self._eraser(folder_path=content_holder.folder_path, prefix=content_holder.prefix)
        result = content_holder.next()
        while result is not None:
            self._writer(path=result[0], content=result[1])
            result = content_holder.next()
