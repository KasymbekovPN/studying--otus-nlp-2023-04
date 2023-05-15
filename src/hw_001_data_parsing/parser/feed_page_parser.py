from bs4 import BeautifulSoup


def parse_by_a_class(content: str, class_: str) -> str:
    # soup = BeautifulSoup(page, 'html.parser')
    # els = soup.find_all('a', class_='x-product-card__link')
    pass


class FeedPageParser:
    def __init__(self,
                 class_: str = 'tm-title__link',
                 parser=parse_by_a_class):
        self._class = class_
        self._parser = parser

    def parse(self, content: str) -> str:
        return self._parser(content, self._class)

    def parse_list(self, content: list) -> list:
        return [self.parse(item) for item in content]

