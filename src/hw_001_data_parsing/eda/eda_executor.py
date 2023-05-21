import re


class ViewCounterConverter:
    def __call__(self, *args, **kwargs) -> float:
        text = kwargs.get('text')
        factor = 1.0
        m = re.match(r'.+K', text)
        if m is not None:
            factor = 1_000.0
        m = re.match(r'.+M', text)
        if m is not None:
            factor = 1_000_000.0

        if factor > 1.0:
            text = text[:-1]

        m = re.match(r'(\d|\.)+', text)
        if m is not None and m.span()[1] == len(text) and len(re.findall(r'\.', text)) < 2:
            return factor * float(text)
        return 0


class MostViewedArticlesTask:
    KEY = 'result.key'

    def __init__(self,
                 top_size=10,
                 converter=ViewCounterConverter()):
        self._top_size = top_size
        self._converter = converter
        self.result = {self.KEY: {}}

    def __call__(self, *args, **kwargs) -> None:
        dataset = kwargs.get('dataset')
        r = {}
        for key, value in dataset.items():
            view_counter = int(self._converter(text=value['view_counter']))
            if view_counter in r:
                r[view_counter].append(key)
            else:
                r[view_counter] = [key]

        # remake
        for view in sorted(list(r.keys()), reverse=True):
            keys = r[view]
            for key in keys:
                self.result[self.KEY][key] = view
                if len(self.result[self.KEY]) == self._top_size:
                    return


class MostFrequentWordsTask:
    KEY = 'result.key'

    def __init__(self,
                 top_size=10,
                 excluded=None):
        self.top_size = top_size
        self.excluded = excluded if excluded is not None else []
        self.result = {self.KEY: {}}

    def __call__(self, *args, **kwargs):
        dataset = kwargs.get('dataset')
        word_counters = {}
        for key, value in dataset.items():
            article = value['article']
            words = article.split(' ')
            for word in words:
                if word in self.excluded:
                    continue
                if word in word_counters:
                    word_counters[word] += 1
                else:
                    word_counters[word] = 1

        by_counters = {}
        for k, v in word_counters.items():
            if v in by_counters:
                by_counters[v].append(k)
            else:
                by_counters[v] = [k]

        for k in sorted(by_counters.keys(), reverse=True):
            keys = by_counters[k]
            for key in keys:
                self.result[self.KEY][key] = k
                if len(self.result[self.KEY]) == self.top_size:
                    return


class EDAExecutor:
    def __init__(self, dataset: dict) -> None:
        self._dataset = dataset
        self._tasks = []

    def add_task(self, task) -> 'EDAExecutor':
        self._tasks.append(task)
        return self

    def start(self):
        for task in self._tasks:
            task(dataset=self._dataset)
