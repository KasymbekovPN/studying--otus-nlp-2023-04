
class Configurator:
    def __init__(self) -> None:
        self._params = {
            'quantity.thread': 8,
            'quantity.feed-pages': 3,
            'prefix.feed-page': 'feed_page_',
            'prefix.article-page': 'article_page_',
            'path.feed-page': '../../output/feed_pages',
            'path.article-page': '../../output/article_pages',
            'path.dataset': '../../output/dataset',
            'source.take-from-dump.feed-page': False,
            'source.take-from-dump.article-page': False,
            'source.take-from-dump.dataset': False,
            'task.most-viewed.top-quantity': 10,
            'task.most-freq-words.top-quantity': 10,
            'task.most-freq-words.excluded': ['в', 'на', 'с'],
            'download.period': 1,
            'download.timeout': 10
        }

    def __call__(self, *args, **kwargs):
        if len(args) == 0 or args[0] not in self._params:
            return None
        return self._params[args[0]]
