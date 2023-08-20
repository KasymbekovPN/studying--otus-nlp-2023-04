import telebot
from flask import Flask, request, Response

import config

bot = telebot.TeleBot(config.token)
app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def index():
    return '<h1>HELLO</h1>'


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
