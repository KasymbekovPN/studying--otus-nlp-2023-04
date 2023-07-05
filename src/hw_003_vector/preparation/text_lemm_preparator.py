import nltk

from nltk.stem import WordNetLemmatizer


class TextLemmPreparator:
    DOWNLOAD_TASKS = ['punkt', 'wordnet']

    def __init__(self, text_key='review'):
        self._text_key = text_key
        [nltk.download(task) for task in self.DOWNLOAD_TASKS]

    def __call__(self, *args, **kwargs):
        datum = kwargs.get('datum')
        tokens = nltk.word_tokenize(datum[self._text_key])
        wnl = WordNetLemmatizer()
        datum[self._text_key] = ' '.join([wnl.lemmatize(word, 'v') for word in tokens])

        return datum
