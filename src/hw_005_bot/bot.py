import os
import flask
import telebot
import torch

from telebot import TeleBot
from queue import Queue
from flask import Flask, request, Response
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration
)

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
    ExecCommandEngineStrategy,
    TaskCommandEngineStrategy,
    HelpCommandEngineStrategy
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
TOKEN_VAR_NAME = 'DEV_TELEGRAM_BOT_TOKEN'
MODEL_CONFIG_PATH = './data/saved'
MODEL_CONFIG_FILES = (
    'config.json',
    'generation_config.json',
    'pytorch_model.bin',
    'special_tokens_map.json',
    'spiece.model',
    'tokenizer_config.json'

)
QUEUE_MAX_SIZE = 100
MODEL_MAX_LENGTH = 300


def create_queue(queue_max_size: int) -> tuple:
    return None, Queue(maxsize=queue_max_size)


def create_users() -> tuple:
    return None, Users()


def create_determinant_chain() -> tuple:
    determinant_chain_ = DeterminantChain([
        SpecificCommandDeterminant('/start', StartCommandEngineStrategy()),
        SpecificCommandDeterminant('/passage', PassageCommandEngineStrategy()),
        SpecificCommandDeterminant('/question', QuestionCommandEngineStrategy()),
        SpecificCommandDeterminant('/exec', ExecCommandEngineStrategy()),
        SpecificCommandDeterminant('/task', TaskCommandEngineStrategy()),
        SpecificCommandDeterminant('/help', HelpCommandEngineStrategy()),
        AnyCommandDeterminant(),
        TextDeterminant()
    ])
    return None, determinant_chain_


def create_bot(token_var_name: str) -> tuple:
    token = os.environ.get(token_var_name)
    token_presented = token is not None
    bot_ = TeleBot(token) if token_presented else None
    error_message_ = None if token_presented else f'{token_var_name}  is absence is environment variables!'

    return error_message_, bot_


def create_model(model_config_path: str, model_config_files: tuple, model_max_length) -> tuple:
    if len(model_config_files) == 0:
        return 'Tuple of config files is empty', None

    absence_files = set()
    for file_name in model_config_files:
        file_path = os.path.join(model_config_path, file_name)
        if not os.path.isfile(file_path):
            absence_files.add(file_path)

    if len(absence_files) > 0:
        return f'Some files does not exist: {absence_files}', None

    if torch.cuda.is_available():
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print(f'We will use the GPU: {torch.cuda.get_device_name(0)}')
        device = torch.device('cuda')
    else:
        print('No GPU available, using the GPU instead.')
        device = torch.device('cpu')

    model_ = Model(T5ForConditionalGeneration.from_pretrained(model_config_path),
                   T5Tokenizer.from_pretrained(model_config_path),
                   device,
                   model_max_length)

    return None, model_


def check_and_create() -> tuple:
    error_messages_ = []

    def enrich_error_messages(em: str | None):
        if em is not None:
            error_messages_.append(em)

    error_message_, queue_ = create_queue(QUEUE_MAX_SIZE)
    enrich_error_messages(error_message_)

    error_message_, users_ = create_users()
    enrich_error_messages(error_message_)

    error_message_, determinant_chain_ = create_determinant_chain()
    enrich_error_messages(error_message_)

    error_message_, bot_ = create_bot(TOKEN_VAR_NAME)
    enrich_error_messages(error_message_)

    error_message_, model_ = create_model(MODEL_CONFIG_PATH, MODEL_CONFIG_FILES, MODEL_MAX_LENGTH)
    enrich_error_messages(error_message_)

    return error_messages_, queue_, users_, determinant_chain_, bot_, model_


def run(engine: Engine,
        host: str,
        port: int):

    app = Flask(__name__)

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
    error_messages, queue, users, determinant_chain, bot, model = check_and_create()
    if len(error_messages) == 0:
        start_pq_task_consumer(queue, model, bot)
        run(Engine(bot, determinant_chain, users, queue), HOST, PORT)
        queue.put(Task.create_shutdown_task())
    else:
        for error_message in error_messages:
            print(f'[ERROR] {error_message}')
    print('It is finished.')
