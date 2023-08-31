import telebot

import config


bot = telebot.TeleBot(config.get_token())


@bot.message_handler(commands=['start', 'help'])
def start_message(message):
    bot.send_message(message.chat.id, 'hello')


@bot.message_handler(content_types=['text'])
def send_text(message):
    text = f'echo: {message.text}'
    bot.send_message(message.chat.id, text)


def run():
    bot.polling(none_stop=True)


if __name__ == '__main__':
    run()
