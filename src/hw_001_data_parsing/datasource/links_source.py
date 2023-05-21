
class PerforatingLinkCreator:
    def __init__(self, links: list):
        self._links = links

    def __call__(self, *args, **kwargs):
        return self._links


class FeedLinksCreator:
    def __init__(self, quantity: int):
        self._quantity = quantity

    def __call__(self, *args, **kwargs):
        return [f'https://habr.com/ru/all/page{i+1}/' for i in range(0, self._quantity)]
