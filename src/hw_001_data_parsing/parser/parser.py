import re

from bs4 import BeautifulSoup
from datetime import datetime


class CollectPageLinksTask:
    KEY = 'links'

    def __init__(self, tag: str, class_: str, base_href='https://habr.com') -> None:
        self.attrs = {self.KEY: []}
        self._tag = tag
        self._class = class_
        self._base_href = base_href

    def execute(self, key: str, soup: 'BeautifulSoup') -> 'CollectPageLinksTask':
        els = soup.find_all(self._tag, self._class)
        self.attrs[self.KEY] += [self._base_href + el.attrs['href'] for el in els]
        print(f'[CollectPageLinksTask] links amount is {len(self.attrs[self.KEY])}')
        return self


class CollectPublishedDatetime:
    def __init__(self, tag: str, class_: str):
        self.attrs = {}
        self._tag = tag
        self._class = class_

    def execute(self, key: str, soup: 'BeautifulSoup') -> 'CollectPublishedDatetime':
        el = soup.find(self._tag, class_=self._class)
        t = el.find('time')
        self.attrs[key] = t.attrs['datetime']
        return self


class CollectViewCounter:
    def __init__(self, tag: str, class_: str):
        self.attrs = {}
        self._tag = tag
        self._class = class_

    def execute(self, key: str, soup: 'BeautifulSoup') -> 'CollectViewCounter':
        el = soup.find(self._tag, class_=self._class)
        self.attrs[key] = el.text
        return self


class CollectArticleTask:
    def __init__(self, tag: str, class_: str):
        self.attrs = {}
        self._tag = tag
        self._class = class_

    def execute(self, key: str, soup: 'BeautifulSoup') -> 'CollectArticleTask':
        el = soup.find(self._tag, class_=self._class)
        text = re.sub(r'<[0-9a-zA-Z="_:/. -]+>', ' ', el.text)
        text = re.sub(r'</[a-zA-Z]+>', ' ', text)
        text = re.sub(r'\W+', ' ', text)
        self.attrs[key] = text.strip()
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


if __name__ == '__main__':
    pass
    # page = '<div><p class="x0 x1 x2"></p></div>'
    #
    # s = BeautifulSoup(page, 'html.parser')
    # els = s.find_all('p', class_='x0')
    # print(els)

    # page = '<div><span class="tm-article-datetime-published"><time datetime="2023-01-23T11:31:09.000Z" title="2023-01-23, ' \
    #        '14:31">23  янв   в 14:31</time></span></div>'
    # soup = BeautifulSoup(page, 'html.parser')
    # r = soup.find('span', class_='tm-article-datetime-published')
    # s = r.next.attrs['datetime']
    # print(s)
    # # print(datetime.strptime(s, '%Y/%m/%d %H:%M:%S.%f'))
    #
    # print(datetime.now())
    # x = datetime(2001, 1, 2, 10, 11, 12)
    # print(x)
