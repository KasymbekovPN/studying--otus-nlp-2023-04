from src.common.base_configurator import BaseConfigurator


class Configurator(BaseConfigurator):
    def __init__(self) -> None:
        params = {
            'path.dataset.original': '../../datasets/IMDB Dataset.csv',
            'path.dataset.prepared': '../../datasets/prepared IMDB Dataset.csv',  # filtered & lemmatized dataset

            'train-test-split.test-size': 0.2,
            'train-test-split.random-stage': 42,

            'vec.tfidf.max-features': 200,
            'vec.tfidf.norm': None,
            'vec.tfidf.max-df': 0.95,
            'vec.tfidf.min-df': 5,
            'vec.tfidf.stop-words': 'english',

            'hyper-params.on': True,
            'hyper-params.cv': 3,
            'hyper-params.scoring': 'accuracy',
            'hyper-params.verbose': 3,
            'hyper-params.jobs': -1,
            'hyper-params.param-grid': {
                'max_depth': [3, None],
                'n_estimators': [10, 100],
            },
            # todo !!! add comment
            # 'hyper-params.param-grid': {
            #     'max_depth': [3, 5, 10, None],
            #     'n_estimators': [10, 100, 200],
            #     'max_features': [1, 5, 10],
            #     'min_samples_leaf': [1, 2, 3],
            #     'min_samples_split': [1, 2, 3]
            # },

            # todo !!! change values cls.<...>
            'cls.max-depth': None,
            'cls.estimators': 500,
            'cls.max-features': 10,
            'cls.min-samples-leaf': 2,
            'cls.min-samples-split': 1
        }
        super().__init__(params)
