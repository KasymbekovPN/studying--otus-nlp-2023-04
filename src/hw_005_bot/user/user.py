

class User:
    INIT_STATE = 0
    WAIT_STATE = 1
    RUN_STATE = 2

    UNDER_MIN_STATE = INIT_STATE - 1
    OVER_MAX_STATE = RUN_STATE + 1

    STATE_NAMES = {INIT_STATE: "INIT", WAIT_STATE: 'WAIT', RUN_STATE: 'RUN'}

    def __init__(self, user_id: int):
        self._id = user_id
        self._state = User.INIT_STATE

    def __repr__(self):
        return f'User {{ id: {self._id}, state: {self.STATE_NAMES[self._state]} }}'

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
