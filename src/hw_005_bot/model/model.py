from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration
)


class Model:
    POS_LABEL = 'верно'
    POS_ANSWER = 'Верно'
    NEG_ANSWER = 'Неверно'

    def __init__(self, model: T5ForConditionalGeneration, tokenizer: T5Tokenizer, device, max_length) -> None:
        self._pos_label = None
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._max_length = max_length
        self._model.to(self._device)
        self._model.eval()

    @property
    def pos_label(self):
        if self._pos_label is None:
            self._pos_label = self._tokenizer(
                self.POS_LABEL,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=2
            )['input_ids'][0][0].item()
        return self._pos_label

    @pos_label.setter
    def pos_label(self, value):
        raise RuntimeError('Setter of pol_label is unsupported.')

    def execute(self, question: str, passage: str) -> str:
        text = self._create_text(question, passage)

        batch = self._tokenize(text)
        batch = {k: v.to(self._device) for k, v in batch.items()}

        tokens = self._model.generate(**batch)
        result = self.pos_label in tokens

        result = self.POS_ANSWER if result else self.NEG_ANSWER
        answer = f'Вопрос:\n\n{question}\n\nПассаж:\n\n{passage}\n\nРезультат: {result}'

        return answer

    def _tokenize(self, text):
        return self._tokenizer(text,
                               return_tensors='pt',
                               padding='max_length',
                               truncation=True,
                               max_length=self._max_length)

    @staticmethod
    def _create_text(question: str, passage: str) -> str:
        return ''.join([question, passage]).lower()
