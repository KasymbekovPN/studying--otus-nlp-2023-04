from telebot import TeleBot
from telebot.types import Update
from queue import Queue

from src.hw_005_bot.message.processing.determinant.chain import DeterminantChain
from src.hw_005_bot.user.users import Users


class Engine:
    def __init__(self,
                 bot: TeleBot,
                 determinant_chain: DeterminantChain,
                 users: Users,
                 task_queue: Queue) -> None:
        self.bot = bot
        self.determinant_chain = determinant_chain
        self.users = users
        self.task_queue = task_queue

    def set_update(self, update: Update):
        if update is None or update.message is None:
            return

        text = update.message.text
        user_id = update.message.from_user.id
        result = self.determinant_chain(text=text)

        result.strategy.execute(user_id, result, self.bot, self.task_queue, self.users, update)
