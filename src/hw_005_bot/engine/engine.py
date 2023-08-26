from telebot import TeleBot
from telebot.types import Update

from src.hw_005_bot.message.processing.determinant.chain import DeterminantChain
from src.hw_005_bot.user.users import Users
from src.hw_005_bot.execution.task_queue import TaskQueue
from src.hw_005_bot.engine.engine_strategies import BaseEngineStrategy


class Engine:
    def __init__(self,
                 bot: TeleBot,
                 determinant_chain: DeterminantChain,
                 users: Users,
                 task_queue: TaskQueue) -> None:
        self.bot = bot
        self.determinant_chain = determinant_chain
        self.users = users
        self.task_queue = task_queue

    def set_update(self, update: Update):
        text = update.message.text
        user_id = update.message.from_user.id
        result = self.determinant_chain(text=text)

        result.strategy.execute(user_id, result, self.bot, self.task_queue, self.users)


        # if result.kind == result.KIND_SPEC_COMMAND:
        #     self.command_strategies[result.command].execute(user_id, result, self.bot, self.task_queue, self.users)
        # elif result.kind == result.KIND_UNKNOWN_COMMAND:
        #     self.unknown_command_strategy.execute(user_id, result, self.bot, self.task_queue, self.users)
        # elif result.kind == result.KIND_TEXT:
        #     self.text_strategy.execute(user_id, result, self.bot, self.task_queue, self.users)
        # else:
        #     BaseEngineStrategy().execute(user_id, result, self.bot, self.task_queue, self.users)


        # # todo it's temp
        # b = BaseEngineStrategy()
        # b.execute()

        # todo del
        # self.bot.send_message(update.message.from_user.id, f'+++ echo: {update.message.text}')

