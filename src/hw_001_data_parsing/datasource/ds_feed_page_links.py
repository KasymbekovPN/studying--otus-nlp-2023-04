

# todo del
def calculate_pack_ranges(configurator: 'Configurator') -> list:
    thread_amount = configurator.thread_amount
    page_amount = configurator.max_feed_page_amount

    div = page_amount // thread_amount
    result = [[i * thread_amount, i * thread_amount + thread_amount] for i in range(0, div)]

    if page_amount % thread_amount != 0:
        result.append([thread_amount * div, page_amount])

    return result

# todo del
def make_replacing(template: str, idx: int) -> str:
    return template.replace('{i}', str(idx + 1))


# todo rename to LinkHolder + use (thread + link) size + link_source as constructor arg
class DSFeedPageLinks:
    TEMPLATE = 'https://habr.com/ru/all/page{i}/'

    def __init__(self,
                 configurator: 'Configurator',
                 range_calculator=calculate_pack_ranges,
                 replacer=make_replacing) -> None:
        self._link_pack = []
        ranges = range_calculator(configurator)
        for pair in ranges:
            self._link_pack.append((replacer(self.TEMPLATE, idx) for idx in range(pair[0], pair[1])))

    @property
    def link_pack(self):
        return self._link_pack

    @link_pack.setter
    def link_pack(self, value):
        raise Exception('[DSFeedPageLinks] setting unsupported')


if __name__ == '__main__':
    from src.hw_001_data_parsing.configurator.configurator import Configurator

    configurator = Configurator()
    configurator.thread_amount = 8
    configurator.max_feed_page_amount = 23

    ds = DSFeedPageLinks(configurator)
    for gen in ds.link_pack:
        print('gen: ', gen)
        for link in gen:
            print('link: ', link)
