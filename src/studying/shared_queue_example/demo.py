import threading
from time import sleep
from random import random
from threading import Thread
from queue import Queue


SIZE = 10


def produce(queue: Queue):
    print(f'\nProducer: running - {threading.current_thread().ident}')
    for _ in range(SIZE):
        value = random()
        sleep(value)
        queue.put(value)
    queue.put(None)
    print('Producer: done')


def consume(queue):
    print(f'\nConsumer: running - {threading.current_thread().ident}')
    while True:
        item = queue.get()
        if item is None:
            break
        print(f'Got {item}')
    print('Consumer: done')


def run0():
    print(f'1: {threading.current_thread().ident}')
    
    queue = Queue()
    consumer = Thread(target=consume, args=(queue,))
    consumer.start()
    producer = Thread(target=produce, args=(queue,))
    producer.start()

    producer.join()
    consumer.join()


def run1():

    print(f'1: {threading.current_thread().ident}')

    queue = Queue(maxsize=100)
    consumer = Thread(target=consume, args=(queue,))
    consumer.start()
    # consumer.join()

    produce(queue)
    print('DONE')


if __name__ == '__main__':
    run0()
    # run1()
