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


def run_load_task(getter=get_page_raw_data, **kwargs) -> dict:
    if not ('link_pack' in kwargs):
        print('[run_load_task] link_pack is absence')
        return {}
    period = kwargs['period'] if 'period' in kwargs else 0.3
    timeout = kwargs['timeout'] if 'timeout' in kwargs else 10
    generator = kwargs['link_pack']

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
                 ds: 'LinksDS',
                 threads_quantity: int) -> None:
        self._ds = ds
        self._threads_quantity = threads_quantity

    def __call__(self, *args, **kwargs) -> dict:
        with ThreadPoolExecutor(self._threads_quantity) as executor:
            result = {}
            futures = []
            for link_pack in self._ds.link_packs:
                futures.append(executor.submit(run_load_task, link_pack=link_pack))
            for future in concurrent.futures.as_completed(futures):
                result = {**result, **future.result()}
        return result


def run():
    from src.hw_001_data_parsing.datasource.links_ds import LinksDS
    from src.hw_001_data_parsing.datasource.links_source import FeedLinksCreator

    threads_quantity = 8
    link_creator = FeedLinksCreator(23)
    ds = LinksDS(threads_quantity, link_creator)

    loader = PageLoader(ds, threads_quantity)
    result = loader()
    print(len(result))
    for k, v in result.items():
        print(k, ' : ', len(v))


if __name__ == '__main__':
    run()
