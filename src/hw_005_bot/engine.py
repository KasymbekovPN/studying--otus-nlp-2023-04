from telebot import TeleBot
from telebot.types import Update


class Engine:
    def __init__(self, bot: TeleBot):
        self._bot = bot

    def set_update(self, update: Update):
        self._bot.send_message(update.message.from_user.id, f'+++ echo: {update.message.text}')

    # todo del
    # def set_command(self, user_id: int, command: str):
    #     print(command, user_id)
    #     # self._bot.send_message(user_id, command)
    #
    # def set_task(self, user_id: int, task: str):
    #     print(task, user_id)
    #     # <
    #     self._bot.send_message(user_id, f'echo: {task}')
