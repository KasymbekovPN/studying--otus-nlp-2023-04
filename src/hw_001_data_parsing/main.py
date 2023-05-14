from src.hw_001_data_parsing.configurator.configurator import Configurator
from src.hw_001_data_parsing.datasource.ds_feed_page_links import DSFeedPageLinks
from src.hw_001_data_parsing.loader.page_loader import PageLoader
from src.hw_001_data_parsing.save.saver import Saver


def run():
    configurator = Configurator()
    ds_feed_page_links = DSFeedPageLinks(configurator)
    loader = PageLoader(ds_feed_page_links, configurator)
    loader_result = loader.execute()
    saver = Saver('../../output/feed_pages', 'feed_page_')
    saver.save_dict(loader_result)


if __name__ == '__main__':
    run()
