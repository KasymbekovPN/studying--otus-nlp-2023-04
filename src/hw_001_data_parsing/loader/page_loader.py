import requests
from time import sleep
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor


def get_page(url: str, timeout: int) -> 'response':
    return requests.Session().get(url=url, timeout=timeout)


def get_page_raw_data(url: str, timeout: int, getter=get_page) -> dict:
    response = getter(url, timeout)
    result = {'status_code': response.status_code, 'raw_text': '', 'url': url}
    if response.status_code == 200:
        result['raw_text'] = response.text
    else:
        print(f'[{response.status_code}] {url}')

    return result


# todo rename link_generator
def run_load_task(getter=get_page_raw_data, **kwargs) -> dict:
    if not ('link_generator' in kwargs):
        print('[run_load_task] link_generator is absence')
        return {}
    period = kwargs['period'] if 'period' in kwargs else 0.3
    timeout = kwargs['timeout'] if 'timeout' in kwargs else 10
    generator = kwargs['link_generator']

    result = {}
    for url in generator:
        page_data = getter(url, timeout)
        status_code = page_data['status_code']
        print(f'[{status_code}] URL : {url}')
        if status_code == 200:
            result[url] = page_data['raw_text']
        sleep(period)

    return result


class PageLoader:
    def __init__(self,
                 ds: 'DSFeedPageLinks',
                 configurator: 'Configurator') -> None:
        self._ds = ds
        self._conf = configurator

    def execute(self) -> dict:
        with ThreadPoolExecutor(self._conf.thread_amount) as executor:
            result = {}
            futures = []
            for generator in self._ds.link_pack:
                futures.append(executor.submit(run_load_task, link_generator=generator))
            for future in concurrent.futures.as_completed(futures):
                result = {**result, **future.result()}
        return result


def run():
    from src.hw_001_data_parsing.configurator.configurator import Configurator
    from src.hw_001_data_parsing.datasource.links_ds import DSFeedPageLinks

    configurator = Configurator()
    configurator.thread_amount = 8
    configurator.max_feed_page_amount = 23

    ds = DSFeedPageLinks(configurator)

    loader = PageLoader(ds, configurator)
    result = loader.execute()
    print(len(result))
    for k, v in result.items():
        print(k, ' : ', len(v))


if __name__ == '__main__':
    run()
