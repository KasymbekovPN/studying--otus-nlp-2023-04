from queue import Queue
from threading import Thread
from telebot import TeleBot

from src.hw_005_bot.model.model import Model
from src.hw_005_bot.execution.task import Task


def consume_pq_task(queue: Queue, model: Model, bot: TeleBot):
    print('\nPQ TASK CONSUMER is started.')

    while True:
        task = queue.get()
        if task.kind == Task.KIND_SHUTDOWN:
            break
        if task.kind == Task.KIND_PQ:
            result = task.get()
            exec_result = model.execute(result['question'], result['passage'])
            bot.send_message(result['user_id'], exec_result)

    print('\nPQ TASK CONSUMER is done.')


def start_pq_task_consumer(queue: Queue, model: Model, bot: TeleBot):
    consumer = Thread(target=consume_pq_task, args=(queue, model, bot))
    consumer.start()
