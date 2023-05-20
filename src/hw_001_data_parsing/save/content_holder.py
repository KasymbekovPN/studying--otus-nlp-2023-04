import hashlib


class HashNameComputer:
    def __call__(self, *args, **kwargs) -> str:
        return kwargs.get('prefix') + str(hashlib.md5(str.encode(kwargs.get('value'))).hexdigest()) + '.htm'


class ContentHolder:
    def next(self) -> tuple:
        raise Exception('[ContentHolder] next unsupported')


class PageContentHolder(ContentHolder):
    def __init__(self,
                 data: dict,
                 folder_path: str,
                 file_prefix: str,
                 name_computer=HashNameComputer()):
        self.folder_path = folder_path
        self.prefix = file_prefix
        self._data = [(f'{folder_path}/{name_computer(prefix=file_prefix, value=v)}', f'{k}\n{v}') for k, v in data.items()]

    def next(self) -> tuple:
        if len(self._data) == 0:
            return None
        return self._data.pop(0)


class DatasetContentHolder(ContentHolder):
    def __init__(self,
                 data: dict,
                 folder_path,
                 file_name='dataset.csv'):
        self.folder_path = folder_path
        self.prefix = file_name
        content = 'id,view_counter,datetime,article\n'
        for k, v in data.items():
            content += f'{k},{v.get("view_counter")},{v.get("datetime")},{v.get("article")}\n'
        self._data = [(f'{folder_path}/{file_name}', content)]

    def next(self) -> tuple:
        if len(self._data) == 0:
            return None
        return self._data.pop(0)