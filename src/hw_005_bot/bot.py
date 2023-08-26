import os
import flask
import telebot

from flask import Flask, request, Response
from src.hw_005_bot.engine.engine import Engine
from src.hw_005_bot.message.processing.determinant.chain import DeterminantChain
from src.hw_005_bot.message.processing.determinant.determinant import (
    SpecificCommandDeterminant,
    AnyCommandDeterminant,
    TextDeterminant
)
from src.hw_005_bot.user.users import Users
from src.hw_005_bot.execution.task_queue import TaskQueue


HOST = 'localhost'
PORT = 5000
FLASK_ABORT_CODE = 403
ROUTE_RULE = '/'
ROUTE_METHODS = ['POST', 'GET']
ENCODING = 'utf-8'


def run(host: str,
        port: int,
        token: str,
        determinant_chain: DeterminantChain,
        users: Users,
        task_queue: TaskQueue):
    bot = telebot.TeleBot(token)
    app = Flask(__name__)
    engine = Engine(bot, determinant_chain, users, task_queue)

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
        # todo del
        # m = Model()
        dc = DeterminantChain([
            SpecificCommandDeterminant('/start'),
            AnyCommandDeterminant(),
            TextDeterminant()
        ])
        us = Users()
        tq = TaskQueue()
        run(HOST, PORT, bot_token, dc, us, tq)
    else:
        print('DEV_TELEGRAM_BOT_TOKEN is absence is environment variables!')
