import copy


class Dataset:
    STRATEGY_NAME_PREFIX = 'strategy_'

    def __init__(self,
                 raw_data: list[dict],
                 headers: set[str],
                 **kwargs) -> None:
        headers = Dataset._prepare_headers(headers)
        self._data = Dataset._prepare_data(raw_data, headers)
        self._strategies = Dataset._prepare_strategies(headers, **kwargs)

    def get(self) -> dict[str, list]:
        return copy.deepcopy(self._data)

    def get_column(self, column: str, strategy=None) -> list | None:
        if strategy is None:
            strategy = column
        if column in self._strategies:
            return [self._strategies.get(strategy)(original) for original in self._data[column]]
        return None

    @classmethod
    def run_default_strategy(cls, original):
        return original

    @classmethod
    def _prepare_headers(cls, headers: set[str]) -> set[str]:
        return {item for item in headers if isinstance(item, str) and item.strip()} \
            if isinstance(headers, set) \
            else set()

    @classmethod
    def _prepare_data(cls, raw_data: list[dict], headers: set[str]) -> dict[str, list]:
        result = {header: [] for header in headers}
        for item in raw_data:
            [result[key].append(value) for key, value in item.items() if headers == item.keys()]

        return result

    @classmethod
    def _prepare_strategies(cls, headers: set[str], **kwargs) -> dict:
        prefix = Dataset.STRATEGY_NAME_PREFIX
        prefix_len = len(prefix)
        result = {key[prefix_len:]: value for key, value in kwargs.items()}
        for header in headers:
            result[header] = Dataset.run_default_strategy

        return result
