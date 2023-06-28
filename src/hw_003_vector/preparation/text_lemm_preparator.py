import nltk

from nltk.stem import WordNetLemmatizer


class TextLemmPreparator:
    DOWNLOAD_TASKS = ['punkt', 'wordnet']

    def __init__(self):
        [nltk.download(task) for task in self.DOWNLOAD_TASKS]

    def __call__(self, *args, **kwargs):
        tokens = nltk.word_tokenize(kwargs.get('text'))
        wnl = WordNetLemmatizer()
        return ' '.join([wnl.lemmatize(word, 'v') for word in tokens])
