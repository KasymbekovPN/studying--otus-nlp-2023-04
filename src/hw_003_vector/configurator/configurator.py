from src.common.base_configurator import BaseConfigurator


class Configurator(BaseConfigurator):
    def __init__(self) -> None:
        params = {
            # todo del
            # 'dataset_path': '../../datasets/IMDB Dataset.csv',
            # 'test_part_percent': 0.3
        }
        super().__init__(params)
