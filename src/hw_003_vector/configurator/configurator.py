from src.common.base_configurator import BaseConfigurator


class Configurator(BaseConfigurator):
    def __init__(self) -> None:
        params = {
            'path.dataset.original': '../../datasets/IMDB Dataset.csv',
            'path.dataset.prepared': '../../datasets/prepared IMDB Dataset.csv',  # filtered & lemmatized dataset

            # todo del
            # 'test_part_percent': 0.3
        }
        super().__init__(params)
