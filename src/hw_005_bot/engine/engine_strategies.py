from telebot import TeleBot
from telebot.types import Update

from src.hw_005_bot.user.user import User
from src.hw_005_bot.user.users import Users
from src.hw_005_bot.execution.task_queue import TaskQueue
from src.hw_005_bot.execution.task import Task


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
        user = users.get_or_add(user_id)
        error_message = self._check_user_status(user)
        if error_message is not None:
            text = error_message
        else:
            task_queue.put(Task.create_pq_task(user.question, user.passage))
            text = 'Задание добавлено в обработку.'

        bot.send_message(user_id, text)

    @staticmethod
    def _check_user_status(user: User) -> str | None:
        error_message = None
        if user.question is None:
            error_message = 'Вопрос не задан'
        if user.passage is None:
            error_message += ', пассаж не задан' if error_message is not None else 'Пассаж не задан'
        if error_message is not None:
            error_message += '.'

        return error_message


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
