from src.common.base_configurator import BaseConfigurator


class Configurator(BaseConfigurator):
    def __init__(self) -> None:
        params = {
            'learning-rate': 0.001,
            'step.epoch-log': 10,
            'quantity.epochs': 2_000,
            'quantity.neurons': [25, 25, 25, 25, 25],
            'size.train.x': 140,
            'size.train.y': 140,
            'size.test.x': 30,
            'size.test.y': 30
        }
        super().__init__(params)
