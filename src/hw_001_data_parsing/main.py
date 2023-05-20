from src.hw_001_data_parsing.configurator.configurator import Configurator
from src.hw_001_data_parsing.datasource.links_ds import LinksDS
from src.hw_001_data_parsing.datasource.links_source import FeedLinksCreator, PerforatingLinkCreator
from src.hw_001_data_parsing.loader.page_loader import PageLoader
from src.hw_001_data_parsing.loader.dump_loader import DumpReader, DumpFilesReader, DatasetFilesReader, DatasetSuitableFileNamesComputer, DumpSuitableFileNamesComputer
from src.hw_001_data_parsing.parser.collector import Collector
from src.hw_001_data_parsing.parser.parser import CollectPageLinksTask, Parser, CollectPublishedDatetimeTask, \
    CollectViewCounterTask, CollectArticleTask
from src.hw_001_data_parsing.save.content_holder import ContentHolder, DatasetContentHolder, PageContentHolder
from src.hw_001_data_parsing.save.saver import Saver


def get_feed_pages(configurator: 'Configurator') -> dict:
    link_creator = FeedLinksCreator(configurator.max_feed_page_amount)
    ds = LinksDS(configurator.thread_amount, link_creator)

    loader = PageLoader(ds, configurator.thread_amount)
    result = loader()

    holder = PageContentHolder(result, configurator.feed_page_folder, configurator.feed_page_prefix)
    Saver().save(holder)

    return result


def get_saved_feed_pages(configurator: 'Configurator') -> dict:
    folder_path = configurator.feed_page_folder
    prefix = configurator.feed_page_prefix
    reader = DumpReader(DumpSuitableFileNamesComputer(folder_path, prefix), DumpFilesReader(folder_path))
    return reader()


def get_article_pages(configurator: 'Configurator', links: list) -> dict:
    link_creator = PerforatingLinkCreator(links)
    ds = LinksDS(configurator.thread_amount, link_creator)

    loader = PageLoader(ds, configurator.thread_amount)
    result = loader()

    holder = PageContentHolder(result, configurator.DEFAULT_ARTICLES_PAGE_FOLDER, configurator.DEFAULT_ARTICLES_PAGE_PREFIX)
    Saver().save(holder)

    return result


def get_saved_article_pages(configurator: 'Configurator') -> dict:
    folder_path = configurator.DEFAULT_ARTICLES_PAGE_FOLDER
    prefix = configurator.DEFAULT_ARTICLES_PAGE_PREFIX
    reader = DumpReader(DumpSuitableFileNamesComputer(folder_path, prefix), DumpFilesReader(folder_path))
    return reader()


def get_dataset(configurator: 'Configurator', article_pages: dict) -> dict:
    collect_published_datetime = CollectPublishedDatetimeTask('span', 'tm-article-datetime-published')
    collect_view_counter_task = CollectViewCounterTask('span', 'tm-icon-counter__value')
    collect_article_task = CollectArticleTask('div', 'article-formatted-body')
    Parser()\
        .add_task(collect_published_datetime)\
        .add_task(collect_view_counter_task)\
        .add_task(collect_article_task)\
        .parse_dict(article_pages)

    dataset = Collector()\
        .add('datetime', collect_published_datetime.attrs)\
        .add('view_counter', collect_view_counter_task.attrs)\
        .add('article', collect_article_task.attrs)\
        .get()

    holder = DatasetContentHolder(dataset, configurator.DEFAULT_DATASET_FOLDER)
    Saver().save(holder)

    return dataset


def get_saved_dataset(configurator: 'Configurator') -> dict:
    folder_path = configurator.DEFAULT_DATASET_FOLDER
    reader = DumpReader(DatasetSuitableFileNamesComputer(folder_path), DatasetFilesReader(folder_path))
    result = {}
    for k, v in reader().items():
        result = v
        break
    return result


FEED_PAGES_FROM_DUMP = True
ARTICLE_PAGES_FROM_DUMP = True
DATASET_FROM_DUMP = True


def run():
    configurator = Configurator()

    feed_pages = get_saved_feed_pages(configurator) if FEED_PAGES_FROM_DUMP else get_feed_pages(configurator)

    if ARTICLE_PAGES_FROM_DUMP:
        article_pages = get_saved_article_pages(configurator)
    else:
        pass
        collect_page_links_task = CollectPageLinksTask('a', 'tm-title__link')
        Parser().add_task(collect_page_links_task).parse_dict(feed_pages)
        article_links = collect_page_links_task.attrs[collect_page_links_task.KEY]
        article_pages = get_article_pages(configurator, article_links)

    dataset = get_saved_dataset(configurator) if DATASET_FROM_DUMP else get_dataset(configurator, article_pages)

    print('')


if __name__ == '__main__':
    run()
