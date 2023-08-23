import os
import flask
import telebot

from flask import Flask, request, Response
from engine import Engine


HOST = 'localhost'
PORT = 5000
FLASK_ABORT_CODE = 403
ROUTE_RULE = '/'
ROUTE_METHODS = ['POST', 'GET']
ENCODING = 'utf-8'
COMMANDS = ['start', 'help', 'state']


def run(host, port, token):
    bot = telebot.TeleBot(token)
    app = Flask(__name__)
    engine = Engine(bot)

    def flask_abort():
        flask.abort(FLASK_ABORT_CODE)

    @app.route(ROUTE_RULE, methods=ROUTE_METHODS)
    def index():
        if request.headers.get('content-type') == 'application/json':
            update = telebot.types.Update.de_json(
                request.stream.read().decode(ENCODING)
            )
            bot.process_new_updates([update])
            print(12345)
            return ''
        flask_abort()
        return Response('ok', status=200) if request.method == 'POST' else ' '

    @bot.message_handler(commands=COMMANDS)
    def handle_command(message):
        # todo: !!!
        #     bot.send_message(message.chat.id, 'hello')
        # <
        # print(message.chat_id)
        bot.send_message(message.chat.id, '!!! command')
        # print(message.chat_id)
        # engine.set_command(message.chat_id, message.text)

    @bot.message_handler(regexp='/.*')
    def handle_unknown_command(message):
        bot.send_message(message.chat.id, f'{message.text} - неизвестная команда!')

    @bot.message_handler(content_types=['text'])
    def handle_text_message(message):
        # todo: !!!
        #     text = f'echo: {message.text}'
        #     bot.send_message(message.chat.id, text)
        #<
        text = f'echo: {message.text}'
        bot.send_message(message.chat.id, text)
        #<
        # engine.set_task(message.chat_id, message.text)

    app.run(host, port)


if __name__ == '__main__':
    bot_token = os.environ.get('DEV_TELEGRAM_BOT_TOKEN')
    if bot_token is not None:
        run(HOST, PORT, bot_token)
    else:
        print('DEV_TELEGRAM_BOT_TOKEN is absence is environment variables!')
