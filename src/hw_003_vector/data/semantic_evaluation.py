from enum import Enum


class SemanticEvaluation(Enum):
    UNKNOWN = 0
    POSITIVE = 1
    NEGATIVE = 2

    @classmethod
    def create(cls, name: str):
        name = name.lower()
        if name == 'positive':
            return SemanticEvaluation.POSITIVE
        elif name == 'negative':
            return SemanticEvaluation.NEGATIVE
        return SemanticEvaluation.UNKNOWN
