import ssl


class SslHolder:
    def __init__(self, verified_context: bool) -> None:
        self._need_reset = False
        self._original = ssl._create_default_https_context
        if not verified_context:
            self._need_reset = True
            ssl._create_default_https_context = ssl._create_unverified_context

    def reset(self) -> None:
        if self._need_reset:
            ssl._create_default_https_context = self._original
