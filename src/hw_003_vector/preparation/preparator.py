
class Preparator:
    def __init__(self, *filters) -> None:
        self._filters = [f for f in filters if callable(f)]

    def __call__(self, *args, **kwargs) -> str | None:
        if len(args) > 0 and isinstance(args[0], str):
            text = args[0]
            for f in self._filters:
                text = f(text)
            return text
        return None
