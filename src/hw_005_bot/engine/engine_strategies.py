import json

from telebot import TeleBot
from telebot.types import Update
from queue import Queue

from src.hw_005_bot.user.user import User
from src.hw_005_bot.user.users import Users
from src.hw_005_bot.execution.task import Task


class BaseEngineStrategy:
    def execute(self,
                user_id: int,
                result,
                bot: TeleBot,
                task_queue: Queue,
                users: Users,
                update: Update):
        bot.send_message(user_id, f'ECHO: {result}')


class StartCommandEngineStrategy(BaseEngineStrategy):
    def execute(self,
                user_id: int,
                result,
                bot: TeleBot,
                task_queue: Queue,
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
                task_queue: Queue,
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
                task_queue: Queue,
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
                task_queue: Queue,
                users: Users,
                update: Update):
        user = users.get_or_add(user_id)
        error_message = self._check_user_status(user)
        if error_message is not None:
            text = error_message
        else:
            task_queue.put(Task.create_pq_task(user.question, user.passage, user_id))
            user.reset_with_state(User.EXEC_STATE)
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


class TaskCommandEngineStrategy(BaseEngineStrategy):
    def execute(self,
                user_id: int,
                result,
                bot: TeleBot,
                task_queue: Queue,
                users: Users,
                update: Update):
        user = users.get_or_add(user_id)
        user.reset_with_state(User.TASK_STATE)
        bot.send_message(user_id, 'Я готов принять задание.')


class HelpCommandEngineStrategy(BaseEngineStrategy):
    def execute(self,
                user_id: int,
                result,
                bot: TeleBot,
                task_queue: Queue,
                users: Users,
                update: Update):

        answer = """Описание команд:
/start

    Стартовая команда, так же используется для сброса 
    состояния.
         
/help

    Команда для вывода справки.

/passage
    
    Команда для перехода в состояние ввода пассажа.
    После использования данной команды последний
    введенный текст будет запоминаться как пассаж. 

/question

    Команда для перехода в состояние ввода вопроса.
    После использования данной команды последний
    введенный текст будет запоминаться как вопрос.

/exec

    Команда для перехода в состояние выполнения 
    задания, состоящего из вопроса и пассажа.

/task

    Команда для введения и выполнения задания,
    которое представляет собой json-строку.
    
    {"question": "...", "passage": "..."}
"""
        bot.send_message(user_id, answer)


class UnknownCommandEngineStrategy(BaseEngineStrategy):
    def execute(self,
                user_id: int,
                result,
                bot: TeleBot,
                task_queue: Queue,
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
                task_queue: Queue,
                users: Users,
                update: Update):
        user = users.get_or_add(user_id)
        if user.state == User.PASSAGE_STATE:
            user.passage = result.text
            text = self._create_passage_question_answer('Задан новый пассаж:', user)
        elif user.state == User.QUESTION_STATE:
            user.question = result.text
            text = self._create_passage_question_answer('Задан новый вопрос:', user)
        elif user.state == User.TASK_STATE:
            text, task = self._prepare_task(result.text)
            if task is not None:
                task_queue.put(Task.create_pq_task(task['question'], task["passage"], user_id))
                user.reset_with_state(User.EXEC_STATE)
        else:
            text = 'Перед вводом вопроса или пассажа введите соответствующие команды ( /question /passage ).'
        bot.send_message(user_id, text)

    @staticmethod
    def _create_passage_question_answer(title: str, user: User):
        return f'{title}\n\nТекущий вопрос:\n\n{user.question}\n\nТекущий пассаж:\n\n{user.passage}'

    @staticmethod
    def _prepare_task(line: str) -> tuple:
        task = None
        try:
            task = json.loads(line)
            text = 'Задание добавлено в обработку.' \
                if 'question' in task and 'passage' in task \
                else 'Задание не содержит question и/или passage.'
        except Exception:
            text = 'Задание имеет неверный формат.'

        return text, task


if __name__ == '__main__':
    pass
