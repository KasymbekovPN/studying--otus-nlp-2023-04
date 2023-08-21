import flask
import telebot
from flask import Flask, request, Response

import config

bot = telebot.TeleBot(config.token)
app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def index():

    def fff():
        flask.abort(403)

    print(request)
    if request.headers.get('content-type') == 'application/json':
        update = telebot.types.Update.de_json(
            request.stream.read().decode('utf-8')
        )
        bot.process_new_updates([update])
        return ''
    else:
        # flask.abort(403)
        fff()
    if request.method == 'POST':
        return Response('ok', status=200)
    else:
        return ' '


@bot.message_handler(commands=['start', 'help'])
def start_message(message):
    bot.send_message(message.chat.id, 'hello')


@bot.message_handler(content_types=['text'])
def send_text(message):
    text = f'echo: {message.text}'
    bot.send_message(message.chat.id, text)


def run():
    # set host and port
    app.run()


if __name__ == '__main__':
    run()
