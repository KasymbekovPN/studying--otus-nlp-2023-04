
# todo del
from time import sleep


class Model:
    def __init__(self):
        pass

    def execute(self, question: str, passage: str) -> bool:
        print(f'{question} -- {passage}')

        # todo del
        sleep(1)

        return False
