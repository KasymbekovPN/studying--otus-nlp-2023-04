

class User:
    INIT_STATE = 0
    QUESTION_STATE = 1
    PASSAGE_STATE = 2
    EXEC_STATE = 3

    UNDER_MIN_STATE = INIT_STATE - 1
    OVER_MAX_STATE = EXEC_STATE + 1

    STATE_NAMES = {
        INIT_STATE: "INIT",
        QUESTION_STATE: 'QUESTION',
        PASSAGE_STATE: 'PASSAGE',
        EXEC_STATE: 'EXEC'
    }

    def __init__(self, user_id: int):
        self._id = user_id
        self._state, self._question, self._passage = None, None, None
        self.reset()

    def __repr__(self):
        return f'User {{ id: {self._id}, state: {self.STATE_NAMES[self._state]}, question: {self._question}, passage: {self._passage} }}'

    def reset(self):
        self._state = User.INIT_STATE
        self._question = None
        self._passage = None

    @property
    def state(self) -> int:
        return self._state

    @state.setter
    def state(self, new_state) -> None:
        self._state = new_state if self.UNDER_MIN_STATE < new_state < self.OVER_MAX_STATE else self.INIT_STATE

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        raise RuntimeError('id setter is unsupported')

    @property
    def question(self):
        return self._question

    @question.setter
    def question(self, value: str):
        self._question = value

    @property
    def passage(self):
        return self._passage

    @passage.setter
    def passage(self, value: str):
        self._passage = value


# todo del
if __name__ == '__main__':
    states = [-2, -1, 0, 1, 2, 3, 4]

    u = User(123)
    print(f'init: {u.state}')
    print()

    for s in states:
        print(f'prev: {u.state}')
        print(f'set: {s}')
        u.state = s
        print(f'new: {u.state}')
        print(u)
        print()
