from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration
)


# todo del
from time import sleep


class Model:
    def __init__(self, model: T5ForConditionalGeneration, tokenizer: T5Tokenizer) -> None:
        self._model = model
        self._tokenizer = tokenizer

    def execute(self, question: str, passage: str) -> bool:
        print(f'{question} -- {passage}')

        # todo del
        sleep(1)

        return False
