

class User:
    INIT_STATE = 0
    WAIT_STATE = 1
    RUN_STATE = 2

    UNDER_MIN_STATE = INIT_STATE - 1
    OVER_MAX_STATE = RUN_STATE + 1

    def __init__(self):
        self._state = User.INIT_STATE

    @property
    def state(self) -> int:
        return self._state

    @state.setter
    def state(self, new_state) -> None:
        self._state = new_state if self.UNDER_MIN_STATE < new_state < self.OVER_MAX_STATE else self.INIT_STATE


# todo del
if __name__ == '__main__':
    states = [-2, -1, 0, 1, 2, 3, 4]

    u = User()
    print(f'init: {u.state}')
    print()

    for s in states:
        print(f'prev: {u.state}')
        print(f'set: {s}')
        u.state = s
        print(f'new: {u.state}')
        print()
