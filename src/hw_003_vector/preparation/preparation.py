
class Preparation:
    def __init__(self, *filters) -> None:
        self._filters = [f for f in filters if callable(f)]

    def __call__(self, *args, **kwargs) -> dict | None:
        if len(args) > 0 and isinstance(args[0], dict):
            datum = args[0]
            for f in self._filters:
                datum = f(datum=datum)
            return datum
        return None
