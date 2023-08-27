
def adapt_shutdown_params(params: list):
    return {}


def adapt_PQ_params(params: list):
    return {'question': params[0], 'passage': params[1], 'user_id': params[2]}


class Task:
    KIND_SHUTDOWN = -1
    KIND_PQ = 0

    UNDER_MIN_KIND = KIND_SHUTDOWN - 1
    OVER_MAX_KIND = KIND_PQ + 1

    ADAPTERS = {
        KIND_SHUTDOWN: adapt_shutdown_params,
        KIND_PQ: adapt_PQ_params
    }

    KIND_NAMES = {
        KIND_SHUTDOWN: 'SHUTDOWN',
        KIND_PQ: 'PQ'
    }

    def __init__(self, kind: int, params: list) -> None:
        self._kind = kind
        self._params = params

    def __repr__(self):
        return f'Task {{ kind: {self.KIND_NAMES[self.kind]}, params: {self.get()} }}'

    @property
    def kind(self) -> int:
        return self._kind

    @kind.setter
    def kind(self, value):
        raise RuntimeError('Setter for kind is unsupported.')

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        raise RuntimeError('Setter for params is unsupported.')

    def get(self) -> dict:
        return self.ADAPTERS[self.kind](self.params)

    @staticmethod
    def create_shutdown_task():
        return Task(Task.KIND_SHUTDOWN, [])

    @staticmethod
    def create_pq_task(question: str, passage: str, user_id: int):
        return Task(Task.KIND_PQ, [question, passage, user_id])


# todo del
if __name__ == '__main__':
    # question = 'some q?'
    # passage = 'some p!'
    #
    # task = Task.create_pq_task(question, passage)
    # print(task)

    # task = Task.create_shutdown_task()
    # print(task)

    pass
