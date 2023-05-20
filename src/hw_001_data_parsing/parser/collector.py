
class Collector:
    def __init__(self):
        self._data = {}

    def add(self, topic: str, part: dict) -> 'Collector':
        for k, v in part.items():
            if k in self._data.keys():
                self._data[k][topic] = v
            else:
                self._data[k] = {topic: v}

        return self

    def get(self) -> dict:
        return self._data
