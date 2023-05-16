from src.hw_001_data_parsing.configurator.configurator import Configurator
from src.hw_001_data_parsing.datasource.ds_feed_page_links import DSFeedPageLinks
from src.hw_001_data_parsing.loader.page_loader import PageLoader
from src.hw_001_data_parsing.loader.dump_loader import DumpLoader
from src.hw_001_data_parsing.save.saver import Saver
from src.hw_001_data_parsing.parser.parser import Parser, CollectPageLinksTask


def get_feed_pages(configurator: 'Configurator'):
    ds_feed_page_links = DSFeedPageLinks(configurator)
    loader = PageLoader(ds_feed_page_links, configurator)
    result = loader.execute()
    saver = Saver(configurator.feed_page_folder, configurator.feed_page_prefix)
    saver.save_dict(result)

    return result


def get_saved_feed_pages(configurator: 'Configurator'):
    loader = DumpLoader(configurator.feed_page_folder, configurator.feed_page_prefix)
    return loader.load()


def run():
    configurator = Configurator()
    # feed_pages = get_feed_pages(configurator)
    feed_pages = get_saved_feed_pages(configurator)

    # for k, v in feed_pages.items():
    #     print(k, ' - ', len(v))

    collect_page_links_task = CollectPageLinksTask('a', 'tm-title__link')
    Parser().add_task(collect_page_links_task).parse_dict(feed_pages)
    # print(collect_page_links_task.attrs[collect_page_links_task.KEY])
    # print(len(collect_page_links_task.attrs[collect_page_links_task.KEY]))


if __name__ == '__main__':
    run()
