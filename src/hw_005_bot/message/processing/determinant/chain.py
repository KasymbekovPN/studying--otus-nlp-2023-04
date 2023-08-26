from src.hw_005_bot.message.processing.determinant.determinant import (
    BaseDeterminant,
    DefaultDeterminant,
    AnyCommandDeterminant,
    SpecificCommandDeterminant,
    TextDeterminant
)


class DeterminantChain:
    def __init__(self, determinants: list[BaseDeterminant]):
        self._determinants = determinants
        self._default_determinant = DefaultDeterminant()

    def __call__(self, *args, **kwargs):
        text = kwargs.get('text') if 'text' in kwargs else None

        for determinant in self._determinants:
            result = determinant(text=text)
            if result is not None:
                return result

        return self._default_determinant(text=text)


# todo del
if __name__ == '__main__':

    texts = [
        None,
        '',
        'start',
        '/help',
        'x123 /help',
        '/help 1234',
        '/start'
    ]

    determinants_ = [
        SpecificCommandDeterminant('/start'),
        AnyCommandDeterminant(),
        TextDeterminant()
    ]
    chain = DeterminantChain(determinants_)

    for text_ in texts:
        print(f'{text_} <> {chain(text=text_)}')

    pass
