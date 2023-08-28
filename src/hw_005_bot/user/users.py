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
