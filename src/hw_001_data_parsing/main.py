from src.hw_001_data_parsing.configurator.configurator import Configurator
from src.hw_001_data_parsing.datasource.links_ds import LinksDS
from src.hw_001_data_parsing.datasource.links_source import FeedLinksCreator, PerforatingLinkCreator
from src.hw_001_data_parsing.loader.page_loader import PageLoader
from src.hw_001_data_parsing.loader.dump_loader import DumpReader, DumpFilesReader, DatasetFilesReader, \
    DatasetSuitableFileNamesComputer, DumpSuitableFileNamesComputer
from src.hw_001_data_parsing.parser.collector import Collector
from src.hw_001_data_parsing.parser.parser import CollectPageLinksTask, Parser, CollectPublishedDatetimeTask, \
    CollectViewCounterTask, CollectArticleTask
from src.hw_001_data_parsing.save.content_holder import DatasetContentHolder, PageContentHolder
from src.hw_001_data_parsing.save.saver import Saver
from src.hw_001_data_parsing.eda.eda_executor import EDAExecutor, MostViewedArticlesTask, MostFrequentWordsTask


def get_feed_pages(configurator: 'Configurator') -> dict:
    link_creator = FeedLinksCreator(configurator('quantity.feed-pages'))
    ds = LinksDS(configurator('quantity.thread'), link_creator)

    loader = PageLoader(ds, configurator('quantity.thread'), configurator('download.period'), configurator('download.timeout'))
    result = loader()

    holder = PageContentHolder(result, configurator('path.feed-page'), configurator('prefix.feed-page'))
    Saver().save(holder)

    return result


def get_saved_feed_pages(configurator: 'Configurator') -> dict:
    folder_path = configurator('path.feed-page')
    reader = DumpReader(DumpSuitableFileNamesComputer(folder_path, configurator('prefix.feed-page')), DumpFilesReader(folder_path))
    return reader()


def get_article_pages(configurator: 'Configurator', feed_pages: dict) -> dict:
    collect_page_links_task = CollectPageLinksTask('a', 'tm-title__link')
    Parser().add_task(collect_page_links_task).parse_dict(feed_pages)
    article_links = collect_page_links_task.attrs[collect_page_links_task.KEY]

    link_creator = PerforatingLinkCreator(article_links)
    ds = LinksDS(configurator('quantity.thread'), link_creator)

    loader = PageLoader(ds, configurator('quantity.thread'), configurator('download.period'), configurator('download.timeout'))
    result = loader()

    holder = PageContentHolder(result, configurator('path.article-page'), configurator('prefix.article-page'))
    Saver().save(holder)

    return result


def get_saved_article_pages(configurator: 'Configurator') -> dict:
    folder_path = configurator('path.article-page')
    prefix = configurator('prefix.article-page')
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

    holder = DatasetContentHolder(dataset, configurator('path.dataset'))
    Saver().save(holder)

    return dataset


def get_saved_dataset(configurator: 'Configurator') -> dict:
    folder_path = configurator('path.dataset')
    reader = DumpReader(DatasetSuitableFileNamesComputer(folder_path), DatasetFilesReader(folder_path))
    result = {}
    for k, v in reader().items():
        result = v
        break
    return result


def print_most_viewed(task):
    print('#'*10, 'THE MOST VIEWED ARTICLES', '#'*10)
    for k, v in task.result[task.KEY].items():
        print(k, ': ', v)


def print_most_frequent(task):
    print('#'*10, 'THE MOST FREQUENT WORDS', '#'*10)
    print('#'*3, 'Excluded words:', ', '.join(task.excluded))
    for k, v in task.result[task.KEY].items():
        print(k, ': ', v)


def run():
    conf = Configurator()

    feed_pages = get_saved_feed_pages(conf) if conf('source_.take-from-dump.feed-page') else get_feed_pages(conf)
    article_pages = get_saved_article_pages(conf) if conf('source_.take-from-dump.article-page') else get_article_pages(conf, feed_pages)
    dataset = get_saved_dataset(conf) if conf('source_.take-from-dump.dataset') else get_dataset(conf, article_pages)

    most_viewed_task = MostViewedArticlesTask(conf('task.most-viewed.top-quantity'))
    most_freq_words = MostFrequentWordsTask(conf('task.most-freq-words.top-quantity'), conf('task.most-freq-words.excluded'))
    EDAExecutor(dataset)\
        .add_task(most_viewed_task)\
        .add_task(most_freq_words)\
        .start()

    print_most_viewed(most_viewed_task)
    print_most_frequent(most_freq_words)


if __name__ == '__main__':
    run()
