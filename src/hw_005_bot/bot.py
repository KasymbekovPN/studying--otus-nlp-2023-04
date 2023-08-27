import os
import flask
import telebot

from telebot import TeleBot
from queue import Queue
from flask import Flask, request, Response
from src.hw_005_bot.engine.engine import Engine
from src.hw_005_bot.message.processing.determinant.chain import DeterminantChain
from src.hw_005_bot.message.processing.determinant.determinant import (
    SpecificCommandDeterminant,
    AnyCommandDeterminant,
    TextDeterminant
)
from src.hw_005_bot.user.users import Users
from src.hw_005_bot.engine.engine_strategies import (
    StartCommandEngineStrategy,
    QuestionCommandEngineStrategy,
    PassageCommandEngineStrategy,
    ExecCommandEngineStrategy
)
from src.hw_005_bot.model.model import Model
from src.hw_005_bot.execution.pq_task_consumer import start_pq_task_consumer
from src.hw_005_bot.execution.task import Task

HOST = 'localhost'
PORT = 5000
FLASK_ABORT_CODE = 403
ROUTE_RULE = '/'
ROUTE_METHODS = ['POST', 'GET']
ENCODING = 'utf-8'


def run(bot: TeleBot,
        host: str,
        port: int,
        determinant_chain: DeterminantChain,
        users: Users,
        task_queue: Queue):
    app = Flask(__name__)
    engine = Engine(bot,
                    determinant_chain,
                    users,
                    task_queue)

    def flask_abort():
        flask.abort(FLASK_ABORT_CODE)

    @app.route(ROUTE_RULE, methods=ROUTE_METHODS)
    def index():
        if request.headers.get('content-type') == 'application/json':
            update = telebot.types.Update.de_json(
                request.stream.read().decode(ENCODING)
            )
            engine.set_update(update)
            return ''
        flask_abort()
        return Response('ok', status=200) if request.method == 'POST' else ' '

    app.run(host, port)


if __name__ == '__main__':
    bot_token = os.environ.get('DEV_TELEGRAM_BOT_TOKEN')
    if bot_token is not None:
        bot = TeleBot(bot_token)

        dc = DeterminantChain([
            SpecificCommandDeterminant('/start', StartCommandEngineStrategy()),
            SpecificCommandDeterminant('/passage', PassageCommandEngineStrategy()),
            SpecificCommandDeterminant('/question', QuestionCommandEngineStrategy()),
            SpecificCommandDeterminant('/exec', ExecCommandEngineStrategy()),
            AnyCommandDeterminant(),
            TextDeterminant()
        ])
        us = Users()
        tq = Queue()

        m = Model()

        start_pq_task_consumer(tq, m, bot)

        # under run-method ???
        run(bot,
            HOST,
            PORT,
            dc,
            us,
            tq
            )

        tq.put(Task.create_shutdown_task())

        print('DONE')
    else:
        print('DEV_TELEGRAM_BOT_TOKEN is absence is environment variables!')
