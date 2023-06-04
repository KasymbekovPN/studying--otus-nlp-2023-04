
class BaseConfigurator:
    def __init__(self, params) -> None:
        self._params = params

    def __call__(self, *args, **kwargs):
        if len(args) == 0 or args[0] not in self._params:
            return None
        return self._params[args[0]]

