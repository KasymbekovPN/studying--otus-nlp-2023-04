from src.hw_005_bot.user.user import User


class Users:
    def __init__(self) -> None:
        self._users = {}

    def __repr__(self):
        return f'Users {self._users}'

    def add(self, user_id: int) -> User | None:
        if user_id in self._users:
            return None
        u = User(user_id)
        self._users[user_id] = u

        return u

    def get_or_add(self, user_id: int) -> User:
        if user_id in self._users:
            return self._users[user_id]
        u = User(user_id)
        self._users[user_id] = u

        return u

    def remove(self, user_id: int) -> User | None:
        if user_id in self._users:
            u = self._users[user_id]
            del self._users[user_id]
            return u
        return None


# todo del
if __name__ == '__main__':

    users = Users()
    print(f'1: {users}')
    print()

    ids = [123, 456, 456, 789]
    # for id_ in ids:
    #     print(f'try add user with id {id_}')
    #     print(f'result: {users.add(id_)}')
    #     print(f'2: {users}')
    #     print()

    # for id_ in ids:
    #     print(f'try get user with id {id_}')
    #     print(f'result: {users.get_or_add(id_)}')
    #     print(f'2: {users}')
    #     print()

    for id_ in ids:
        users.get_or_add(id_)
    print(f'2: {users}')
    print()
    for id_ in ids:
        print(f'try remove user with id {id_}')
        print(f'result: {users.remove(id_)}')
        print(f'3: {users}')
        print()

    pass
