from telebot import TeleBot

from src.hw_005_bot.user.users import Users
# todo del
from src.hw_005_bot.user.user import User
from src.hw_005_bot.execution.task_queue import TaskQueue
# from src.hw_005_bot.message.processing.determinant.result import DeterminationResult


class BaseEngineStrategy:
    def execute(self,
                user_id: int,
                result,
                bot: TeleBot,
                task_queue: TaskQueue,
                users: Users):
        bot.send_message(user_id, f'ECHO: {result}')


class StartCommandEngineStrategy(BaseEngineStrategy):
    def execute(self,
                user_id: int,
                result,
                bot: TeleBot,
                task_queue: TaskQueue,
                users: Users):
        # user = users.get_or_add(user_id)
        # print(user)
        # user1 = users.get_or_add(user_id)
        # print(user1)
        #
        # # todo del
        bot.send_message(user_id, f'start ECHO: {result}')


class QuestionCommandEngineStrategy(BaseEngineStrategy):
    def execute(self,
                user_id: int,
                result,
                bot: TeleBot,
                task_queue: TaskQueue,
                users: Users):
        bot.send_message(user_id, f'question ECHO: {result}')


class PassageCommandEngineStrategy(BaseEngineStrategy):
    def execute(self,
                user_id: int,
                result,
                bot: TeleBot,
                task_queue: TaskQueue,
                users: Users):
        bot.send_message(user_id, f'passage ECHO: {result}')


class ExecCommandEngineStrategy(BaseEngineStrategy):
    def execute(self,
                user_id: int,
                result,
                bot: TeleBot,
                task_queue: TaskQueue,
                users: Users):
        bot.send_message(user_id, f'exec ECHO: {result}')


class UnknownCommandEngineStrategy(BaseEngineStrategy):
    def execute(self,
                user_id: int,
                result,
                bot: TeleBot,
                task_queue: TaskQueue,
                users: Users):
        bot.send_message(user_id, f'unknown ECHO: {result}')


class TextEngineStrategy(BaseEngineStrategy):
    def execute(self,
                user_id: int,
                result,
                bot: TeleBot,
                task_queue: TaskQueue,
                users: Users):
        bot.send_message(user_id, f'text ECHO: {result}')


if __name__ == '__main__':
    pass
