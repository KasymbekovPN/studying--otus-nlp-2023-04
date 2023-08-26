from telebot import TeleBot
from telebot.types import Update

from src.hw_005_bot.user.user import User
from src.hw_005_bot.user.users import Users
from src.hw_005_bot.execution.task_queue import TaskQueue


class BaseEngineStrategy:
    def execute(self,
                user_id: int,
                result,
                bot: TeleBot,
                task_queue: TaskQueue,
                users: Users,
                update: Update):
        bot.send_message(user_id, f'ECHO: {result}')


class StartCommandEngineStrategy(BaseEngineStrategy):
    def execute(self,
                user_id: int,
                result,
                bot: TeleBot,
                task_queue: TaskQueue,
                users: Users,
                update: Update):
        user = users.get_or_add(user_id)
        user.reset()
        bot.send_message(user_id, 'Привет, я - перезапущен.')


class QuestionCommandEngineStrategy(BaseEngineStrategy):
    def execute(self,
                user_id: int,
                result,
                bot: TeleBot,
                task_queue: TaskQueue,
                users: Users,
                update: Update):
        user = users.get_or_add(user_id)
        user.state = User.QUESTION_STATE
        bot.send_message(user_id, 'Я готов принять вопрос.')


class PassageCommandEngineStrategy(BaseEngineStrategy):
    def execute(self,
                user_id: int,
                result,
                bot: TeleBot,
                task_queue: TaskQueue,
                users: Users,
                update: Update):
        user = users.get_or_add(user_id)
        user.state = User.PASSAGE_STATE
        bot.send_message(user_id, 'Я готов принять пассаж.')


class ExecCommandEngineStrategy(BaseEngineStrategy):
    def execute(self,
                user_id: int,
                result,
                bot: TeleBot,
                task_queue: TaskQueue,
                users: Users,
                update: Update):
        # todo impl
        bot.send_message(user_id, f'exec ECHO: {result}')


class UnknownCommandEngineStrategy(BaseEngineStrategy):
    def execute(self,
                user_id: int,
                result,
                bot: TeleBot,
                task_queue: TaskQueue,
                users: Users,
                update: Update):
        user = users.get_or_add(user_id)
        user.reset()
        bot.send_message(user_id, f'Мне неизвестна команда {result.text}.')


class TextEngineStrategy(BaseEngineStrategy):
    def execute(self,
                user_id: int,
                result,
                bot: TeleBot,
                task_queue: TaskQueue,
                users: Users,
                update: Update):
        user = users.get_or_add(user_id)
        if user.state == User.PASSAGE_STATE:
            user.passage = result.text
            text = self._create_passage_question_answer('Задан новый пассаж:', user)
        elif user.state == User.QUESTION_STATE:
            user.question = result.text
            text = self._create_passage_question_answer('Задан новый вопрос:', user)
        else:
            text = 'Перед вводом вопроса или пассажа введите соответствующие команды ( /question /passage ).'
        bot.send_message(user_id, text)

    @staticmethod
    def _create_passage_question_answer(title: str, user: User):
        return f'{title}\n\nТекущий вопрос:\n\n{user.question}\n\nТекущий пассаж:\n\n{user.passage}'


if __name__ == '__main__':
    pass
