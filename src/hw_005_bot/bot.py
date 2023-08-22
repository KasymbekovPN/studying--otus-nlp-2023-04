import os
import flask
import telebot

from flask import Flask, request, Response


HOST = 'localhost'
PORT = 5000
FLASK_ABORT_CODE = 403
ROUTE_RULE = '/'
ROUTE_METHODS = ['POST', 'GET']
ENCODING = 'utf-8'
COMMANDS = ['start']


def run(host, port, token):
    bot = telebot.TeleBot(token)
    app = Flask(__name__)

    def flask_abort():
        flask.abort(FLASK_ABORT_CODE)

    @app.route(ROUTE_RULE, methos=ROUTE_METHODS)
    def index():
        if request.headers.get('content-type') == 'application/json':
            update = telebot.types.Update.de_json(
                request.stream.read().decode(ENCODING)
            )
            bot.process_new_updates([update])
            return ''
        flask_abort()
        return Response('ok', status=200) if request.method == 'POST' else ' '

    @bot.message_handler(commands=COMMANDS)
    def handle_command_message(message):
        # todo: !!!
        #     bot.send_message(message.chat.id, 'hello')
        pass

    @bot.message_handler()
    def handle_any_message(message):
        # todo: !!!
        #     bot.send_message(message.chat.id, 'hello')
        pass

    @bot.message_handler(content_types=['text'])
    def handle_text_message(message):
        # todo: !!!
        #     text = f'echo: {message.text}'
        #     bot.send_message(message.chat.id, text)
        pass

    app.run(host, port)


if __name__ == '__main__':
    bot_token = os.environ.get('DEV_TELEGRAM_BOT_TOKEN')
    if bot_token is not None:
        run(HOST, PORT, bot_token)
    else:
        print('DEV_TELEGRAM_BOT_TOKEN is absence is environment variables!')
