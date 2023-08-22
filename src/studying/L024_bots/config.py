import os


def get_token():
    return os.environ.get('DEV_TELEGRAM_BOT_TOKEN')
