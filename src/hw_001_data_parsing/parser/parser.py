from bs4 import BeautifulSoup


class CollectPageLinksTask:
    def __init__(self, tag: str, class_: str) -> None:
        self.attrs = {}
        self._tag = tag
        self._class = class_

    def execute(self, soup: 'BeautifulSoup', key='links'):
        els = soup.find_all(self._tag, self._class)
        self.attrs[key] = [el.attrs['href'] for el in els]
        return self


class Parser:
    def __init__(self):
        self._tasks = []

    def add_task(self, task) -> 'Parser':
        self._tasks.append(task)
        return self

    def parse(self, content: str, key: str):
        pass

    def parse_dict(self, content: dict):
        pass

#  todo del
# if __name__ == '__main__':
#     from src.hw_001_data_parsing.loader.dump_loader import DumpLoader
#     loader = DumpLoader('../../../output/feed_pages', 'feed_page_')
#     result = loader.load()
#
#     # for k, v in result.items():
#     #     soup = BeautifulSoup(v, 'html.parser')
#     #     # print(soup)
#     #     els = soup.find_all('a', class_='tm-title__link')
#     #     print(len(els))
#     #     for el in els:
#     #         print(10*'-')
#     #         print(el)
#     #         print(el.attrs['href'])
#     #         print(el.next)
#     #         print(el.next.text)
#
#     # els = soup.find_all('a', class_='x-product-card__link')
#
#     # tm-title__link
#
#     # saver = Saver('../../../output/feed_pages', 'prefix_')
#     pass
