
class Configurator:
    # todo ??? DEFAULT_OUTPUT_FOLDER_PREFIX + check attributes usage
    DEFAULT_OUTPUT_FOLDER_PREFIX = 'result_'
    DEFAULT_THREAD_AMOUNT = 8
    DEFAULT_MAX_FEED_PAGE_AMOUNT = 5
    DEFAULT_FREQ_TOP = 10
    DEFAULT_REQUEST_PERIOD = 0.3
    DEFAULT_GET_TIMEOUT = 10
    DEFAULT_FEED_PAGE_FOLDER = '../../output/feed_pages'
    DEFAULT_FEED_PAGE_PREFIX = 'feed_page_'

    def __init__(self) -> None:
        self._output_folder_prefix = self.DEFAULT_OUTPUT_FOLDER_PREFIX
        self._thread_amount = self.DEFAULT_THREAD_AMOUNT
        self._max_feed_page_amount = self.DEFAULT_MAX_FEED_PAGE_AMOUNT
        self._freq_top = self.DEFAULT_FREQ_TOP
        self._request_period = self.DEFAULT_REQUEST_PERIOD
        self._get_timeout = self.DEFAULT_GET_TIMEOUT
        self._feed_page_folder = self.DEFAULT_FEED_PAGE_FOLDER
        self._feed_page_prefix = self.DEFAULT_FEED_PAGE_PREFIX

    @property
    def output_folder_prefix(self):
        return self._output_folder_prefix

    @output_folder_prefix.setter
    def output_folder_prefix(self, value):
        self._output_folder_prefix = value

    @property
    def thread_amount(self):
        return self._thread_amount

    @thread_amount.setter
    def thread_amount(self, value):
        self._thread_amount = value

    @property
    def max_feed_page_amount(self):
        return self._max_feed_page_amount

    @max_feed_page_amount.setter
    def max_feed_page_amount(self, value):
        self._max_feed_page_amount = value

    @property
    def freq_top(self):
        return self._freq_top

    @freq_top.setter
    def freq_top(self, value):
        self._freq_top = value

    @property
    def request_period(self):
        return self._request_period

    @request_period.setter
    def request_period(self, value):
        self._request_period = value

    @property
    def get_timeout(self):
        return self._get_timeout

    @get_timeout.setter
    def get_timeout(self, value):
        self._get_timeout = value

    @property
    def feed_page_folder(self):
        return self._feed_page_folder

    @feed_page_folder.setter
    def feed_page_folder(self, value):
        self._feed_page_folder = value

    @property
    def feed_page_prefix(self):
        return self._feed_page_prefix

    @feed_page_prefix.setter
    def feed_page_prefix(self, value):
        self._feed_page_prefix = value
