from bs4 import BeautifulSoup


class CollectPageLinksTask:
    KEY = 'links'

    def __init__(self, tag: str, class_: str) -> None:
        self.attrs = {}
        self._tag = tag
        self._class = class_

    def execute(self, key: str, soup: 'BeautifulSoup') -> 'CollectPageLinksTask':
        els = soup.find_all(self._tag, self._class)
        if self.KEY in self.attrs:
            self.attrs[self.KEY] += [el.attrs['href'] for el in els]
        else:
            self.attrs[self.KEY] = [el.attrs['href'] for el in els]
        print(f'[CollectPageLinksTask] links amount is {len(self.attrs[self.KEY])}')
        return self


class Parser:
    def __init__(self):
        self._tasks = []

    def add_task(self, task) -> 'Parser':
        self._tasks.append(task)
        return self

    def parse(self, key: str, content: str) -> 'Parser':
        soup = BeautifulSoup(content, 'html.parser')
        for task in self._tasks:
            task.execute(key, soup)
        return self

    def parse_dict(self, content: dict) -> 'Parser':
        for k, v in content.items():
            self.parse(k, v)
        return self
