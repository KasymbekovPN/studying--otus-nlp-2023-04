from src.common.base_configurator import BaseConfigurator


class Configurator(BaseConfigurator):
    def __init__(self) -> None:
        params = {
            'path.dataset.original': '../../Datasets/IMDB Dataset.csv',
            'path.dataset.prepared': '../../Datasets/prepared IMDB Dataset.csv',  # filtered & lemmatized dataset

            'train-test-split.test-size': 0.2,
            'train-test-split.random-stage': 42,

            'vec.tfidf.max-features': 300,
            'vec.tfidf.norm': None,
            'vec.tfidf.max-df': 0.95,
            'vec.tfidf.min-df': 5,
            'vec.tfidf.stop-words': 'english',

            'hyper-params.on': False,
            'hyper-params.cv': 3,
            'hyper-params.scoring': 'accuracy',
            'hyper-params.verbose': 3,
            'hyper-params.jobs': -1,
            'hyper-params.param-grid': {
                'max_depth': [3, None],
                'n_estimators': [10, 100, 200],
            },
            'cls.max-depth': 3,
            'cls.estimators': 50,
        }
        super().__init__(params)
