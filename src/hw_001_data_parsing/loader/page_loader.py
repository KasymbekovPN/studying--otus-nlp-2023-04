import requests
from time import sleep
from concurrent.futures import ThreadPoolExecutor


def get_page(url: str, timeout: int) -> 'response':
    return requests.Session().get(url=url, timeout=timeout)


# todo must return either text or None
def get_page_raw_data(url: str, timeout: int, getter=get_page) -> dict:
    response = getter(url, timeout)
    result = {'status_code': response.status_code, 'raw_text': '', 'url': url}
    if response.status_code == 200:
        result['raw_text'] = response.text
    else:
        print(f'[{response.status_code}] {url}')

    return result


def run_load_task(getter=get_page_raw_data, **kwargs) -> list:
    if not ('link_generator' in kwargs):
        print('[run_load_task] link_generator is absence')
        return []
    period = kwargs['period'] if 'period' in kwargs else 0.3
    timeout = kwargs['timeout'] if 'timeout' in kwargs else 10
    generator = kwargs['link_generator']

    result = []
    for url in generator:
        page_data = getter(url, timeout)
        if page_data['status_code'] == 200:
            result.append(page_data['raw_text'])
        sleep(period)

    return result



# todo del
# def get_wiki_page_existence(wiki_page_url, timeout=10):
#     session = requests.Session()
#     response = session.get(url=wiki_page_url, timeout=timeout)
#
#     # response = requests.get(url=wiki_page_url, timeout=timeout)
#
#     page_status = "unknown"
#     if response.status_code == 200:
#         page_status = "exists"
#     elif response.status_code == 404:
#         page_status = "does not exist"
#
#     print(len(response.text))
#     return wiki_page_url + " - " + page_status
#
#
# wiki_page_urls = [
#     "https://en.wikipedia.org/wiki/Ocean",
#     "https://en.wikipedia.org/wiki/Island",
#     "https://en.wikipedia.org/wiki/this_page_does_not_exist",
#     "https://en.wikipedia.org/wiki/Shark",
#     "https://habr.com/ru/all/page1/"
# ]
#
#
# def run():
#
#     x = (x for x in range(0, 10))
#
#     import concurrent.futures
#
#     # with concurrent.futures.ThreadPoolExecutor() as executor:
#     #     futures = []
#     #     for url in wiki_page_urls:
#     #         futures.append(executor.submit(run_load_task,link_generator=x))
#     #     for future in concurrent.futures.as_completed(futures):
#     #         print(future.result())
#     with ThreadPoolExecutor() as executor:
#         futures = []
#         for url in wiki_page_urls:
#             futures.append(executor.submit(get_wiki_page_existence, wiki_page_url=url))
#         for future in concurrent.futures.as_completed(futures):
#             print(future.result())
#
#
# if __name__ == '__main__':
#     run()
