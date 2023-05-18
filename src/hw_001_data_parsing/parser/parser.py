from bs4 import BeautifulSoup
from datetime import datetime


class CollectPageLinksTask:
    KEY = 'links'

    def __init__(self, tag: str, class_: str, base_href='https://habr.com') -> None:
        self.attrs = {self.KEY: []}
        self._tag = tag
        self._class = class_
        self._base_href = base_href

    def execute(self, key: str, soup: 'BeautifulSoup') -> 'CollectPageLinksTask':
        els = soup.find_all(self._tag, self._class)
        self.attrs[self.KEY] += [self._base_href + el.attrs['href'] for el in els]
        print(f'[CollectPageLinksTask] links amount is {len(self.attrs[self.KEY])}')
        return self


class CollectPublishedDatetime:
    def __init__(self, tag: str, class_: str):
        self.attrs = {}
        self._tag = tag
        self._class = class_

    def execute(self, key: str, soup: 'BeautifulSoup') -> 'CollectPageLinksTask':
        el = soup.find(self._tag, class_=self._class)
        t = el.find('time')
        self.attrs[key] = t.attrs['datetime']



# <span class="tm-icon-counter__value">2.1K</span>

# <div class="article-formatted-body article-formatted-body article-formatted-body_version-2"><p>Привет, %username%! Меня зовут Кирилл Фурзанов, я Data Scientist в Сбере, участник&nbsp;<a href="https://habr.com/ru/users/NewTechAudit/posts/" rel="noopener noreferrer nofollow">профессионального сообщества NTA</a>. При формировании витрин данных и датасетов в экосистеме Hadoop одним из важных вопросов является выбор оптимального способа хранения данных в hdfs. Рассмотрим один из важных вопросов при создании витрины – выбор соответствующего формата файла для хранения.</p><figure class="full-width "><img src="https://habrastorage.org/getpro/habr/upload_files/f5a/0bc/288/f5a0bc2880d7bfb2889ca5a9ade49dc5.jpg" width="1280" height="720"><figcaption></figcaption></figure><p>Решением данной задачи я занимался при формировании датасета по транзакциям клиентов за последние полгода. Попробую, с точки зрения теории, определить наилучший формат для хранения данных нашей задачи. Первоначально опишем существующие форматы файлов. Hdfs поддерживает несколько основных типов форматов файлов для хранения, которые принято разделять на 2 группы:</p><p>Строковые форматы: csv, json, avro.</p><p>Колоночные форматы: rc, orc, parquet.</p><p><em>Строковые</em> типы данных, такие как csv, json, avro сохраняют данные в строки. Данный тип позволяет достаточно быстро записывать данные в витрину, а такой формат, как avro позволяет так же динамически изменять схему данных. Однако, задачи по чтению и фильтрации данных строковых форматов требуют большего времени, чем в колоночных форматах из-за самого алгоритма чтения и фильтрации данных. Данный алгоритм можно описать в 3 этапа:</p><p>1) выбирается одна отдельная строка;</p><p>2) производится разделение этой строки на столбцы;</p><p>3) производится фильтрация данных и применение функций к соответствующим строкам.</p><p>Кроме того, строковые форматы зачастую занимают значительно большее место в хранилище, чем колоночные форматы, ввиду их плохой сжимаемости. Кратко опишем вышеперечисленные строковые форматы. </p><p>CSV – &nbsp;распространенный формат данных, используемый зачастую для передачи информации между различными хранилищами. Схема данных в таком формате статична и не подлежит изменению.</p><p>JSON – хранение данных на hdfs в формате json несколько отличается от привычного json-файла тем, что каждая строка представляет собой отдельный json-файл. </p><p>Примером данного формата файла может послужить следующий набор данных:</p><pre><code>{id: 1, name: ‘Иван’, surname: ‘Иванов’, second_name: ‘Иванович’}
# {id: 2, name: ‘Петр’, surname: ‘Петров’, second_name: ‘Петрович’}</code></pre><p>Как видно из примера, основным отличием данного формата от формата CSV является его разделимость и то, что он содержит метаданные вместе с данными, что позволяет изменять исходную схему данных. Так же данный формат файлов позволяет хранить сложные структуры данных в колонках, что является еще одним отличием данного формата перед CSV форматом. К недостаткам данного формата можно отнести так же плохую сжимаемость в блоках данных и значительные затраты на хранение данного типа файлов (даже по сравнению с форматом CSV из-за необходимости дублирования метаданных для каждой строки).</p><p>Наибольшую же популярность среди строковых форматов хранения данных занимает формат avro. Данный формат представляет собой контейнер, содержащий в себе заголовок и один или несколько блоков с данными:</p><figure class="full-width "><img src="https://habrastorage.org/getpro/habr/upload_files/499/04e/391/49904e391a3efbf7f915a70bfe67b6d9.png" width="624" height="252"><figcaption></figcaption></figure><p>Заголовок состоит из:</p><ul><li><p>ASCII слова ‘Obj1’.</p></li><li><p>Метаданных файла, включая определение схемы.</p></li><li><p>16-байтного случайного слова (маркера синхронизации).</p></li></ul><p>Блоки данных в свою очередь состоят из:</p><ul><li><p>Числа объектов в блоке.</p></li><li><p>Размера серриализованых объектов в блоке.</p></li><li><p>Сами данные, представленные в двоичном виде с определенным типом сжатия.</p></li><li><p>16-байтного случайного слова (маркера синхронизации).</p></li></ul><p>Основным преимуществом данного формата являются его сжимаемость, возможность изменения схемы данных. К минусам можно отнести то, что этот формат отсутствует из коробки и для возможности чтения и записи необходимо устанавливать внешний компонент Avro.</p><p><em>Колоночные</em> же форматы напротив занимают значительно большее время для записи данных, однако же они значительно быстрее решают задачи чтения и фильтрации данных, занимая при этом значительно меньшее пространство на дисках, по сравнению со строковыми форматами. Наиболее распространенными колоночными форматами являются форматы orc и parquet.</p><p>ORC (Optimized Record Columnar File) – &nbsp;колоночный формат данных, разделяющий исходный набор данных на полосы, размером в 250M каждая. Колонки в таких полосках разделены друг от друга, обеспечивая избирательное чтение, что увеличивает скорость обработки такой полоски. Кроме того, данные полоски хранят в себе индексы и предикаты, что в свою очередь обеспечивает еще большую скорость чтения и фильтрации данных. Ниже приведена схема структуры ORC файла.</p><figure class="full-width "><img src="https://habrastorage.org/getpro/habr/upload_files/59c/781/ae7/59c781ae7c0bd6f4065a594e3e86df52.png" width="624" height="379"><figcaption></figcaption></figure><p>Parquet – это бинарный колоночно-ориентированный формат данных. Данный формат имеет сложную структуру данных, которая разделяется на 3 уровня:</p><ul><li><p>Row-group – логическое горизонтальное разбиение данных на строки, состоящие из фрагментов каждого столбца в наборе данных.</p></li><li><p>Column chunk – фрагмент конкретного столбца.</p></li><li><p>Page – Содержит в себе наборы row-group. Данные страницы разграничиваются между собой с помощью header и footer. Header содержит волшебное число PAR1, определяющее начало страницы. Footer содержит стартовые координаты каждого из столбцов, версию формата, схему данных, длину метаданных (4 байта), и волшебное число PAR1 как флаг окончания.</p></li></ul><p>Более подробно структура файла parquet представлена на следующей схеме:</p><figure class="full-width "><img src="https://habrastorage.org/getpro/habr/upload_files/3b9/0b2/7f7/3b90b27f764df542a631dfbe54705ec5.png" width="1100" height="994"><figcaption></figcaption></figure><p>Выбор определенного типа и группы форматов файлов во многом зависит от решаемой задачи. </p><p>Мой датасет состоит в основном из простейших типов данных, таких как числа, дата со временем и небольших строковых данных, в связи с чем каких-либо ограничений на выбор формата для хранения у нас нет, и основным критерием будет обеспечение наилучшего быстродействия для операций чтения и фильтрации данных, при этом обеспечивая минимальный занимаемый размер данных. В связи с этим я построил краткую матрицу, описывающую качественные характеристики вышеизложенных форматов на основе заданных критериев:</p><div><div class="table"><div class="table table_wrapped"><table><tbody><tr><td><p align="left"><strong>Формат</strong></p></td><td><p align="left"><strong>Чтение   и фильтрация данных</strong></p></td><td><p align="left"><strong>Подходит   для длительного хранения большого объема данных</strong></p></td><td><p align="left"><strong>Сжимаемость</strong></p></td></tr><tr><td><p align="left">csv</p></td><td><p align="left">Медленное чтение   и фильтрация</p></td><td><p align="left">Нет</p></td><td><p align="left">Не сжимаемый   формат</p></td></tr><tr><td><p align="left">json</p></td><td><p align="left">Медленное   чтение и фильтрация</p></td><td><p align="left">Нет</p></td><td><p align="left">Не сжимаемый   формат</p></td></tr><tr><td><p align="left">avro</p></td><td><p align="left">Медленное   чтение и фильтрация</p></td><td><p align="left">Нет</p></td><td><p align="left">Сжимаемый   формат</p></td></tr><tr><td><p align="left">orc</p></td><td><p align="left">Достаточная   скорость чтения и фильтрации</p></td><td><p align="left">Да</p></td><td><p align="left">Сжимаемый формат</p><p align="left">(Наилучше сжатие)</p></td></tr><tr><td><p align="left">parquet</p></td><td><p align="left">Достаточная   скорость чтения и фильтрации</p></td><td><p align="left">Да</p></td><td><p align="left">Сжимаемый формат</p></td></tr></tbody></table></div></div></div><p>Формат AVRO далее не будет рассматриваться, так как не поддерживается на нашем кластере. Ниже приведена матрица сопоставления количественных характеристик данных форматов файлов на реализованном датасете, который содержит около 100 млн. строк различных простых форматов:</p><div><div class="table"><div class="table table_wrapped"><table><tbody><tr><td><p align="left"><strong>Формат</strong></p></td><td><p align="left"><strong>Занимаемый   объем </strong></p></td><td><p align="left"><strong>Скорость   чтения данных без фильтра (avg)</strong></p></td><td><p align="left"><strong>Скорость   чтения данных с фильтром (avg)</strong></p></td></tr><tr><td><p align="left">csv</p></td><td><p align="left">36GB</p></td><td><p align="left">23 сек.</p></td><td><p align="left">12 сек.</p></td></tr><tr><td><p align="left">json</p></td><td><p align="left">84GB</p></td><td><p align="left">27 сек.</p></td><td><p align="left">25 сек.</p></td></tr><tr><td><p align="left">orc</p></td><td><p align="left">4.9GB</p></td><td><p align="left">4 сек.</p></td><td><p align="left">4.4 сек.</p></td></tr><tr><td><p align="left">parquet</p></td><td><p align="left">5.8GB</p></td><td><p align="left">6 сек.</p></td><td><p align="left">5.9 сек.</p></td></tr></tbody></table></div></div></div><p>Для большей достоверности результатов расчета были применены следующие условия:</p><p>1)&nbsp;Запуск задач производился с одинаковой конфигурацией для каждой из job-ов.</p><p>2)&nbsp;Количество запусков тестового запроса было равно 20.</p><p>3)&nbsp;Одинаковые запросы для каждого формата.</p><p>4)&nbsp;Задачи отрабатывали на свободном кластере.</p><p>5) Время чтения данных учитывалось без времени, затрачиваемом на запуск и остановку job-а.</p><p>Как видно из приведенных таблиц, для решения нашей задачи наиболее подходящим форматом является ORC, так как он предоставлял наилучшее сжатие данных при этом давая оптимальную производительность выполнения запросов на чтение данных.</p><p>Ниже приведу простой код создания таблиц с определенным форматом посредством hiveQL запроса:</p><pre><code>CREATE TABLE tbl_name (id int, ….)
# STORED AS [ORC|JSON|CSV|PARQUET|AVRO]</code></pre><p>Или:</p><pre><code>CREATE TABLE tbl_name
# STORED AS [ORC|JSON|CSV|PARQUET|AVRO]
# As select * from another_table</code></pre><p>И запрос на языке Spark на примере создания таблицы формата parquet:</p><pre><code>Spark_df.write.format(‘parquet’).mode(‘overwrite’).saveAsTable(‘tablename’)</code></pre><p>В результате были рассмотрены различные форматы данных, кратко описаны их плюсы и минусы и приведен код для создания таблиц с выбором соответствующего формата. Надеюсь, эта публикация была для вас полезна.</p><p></p></div>


class Parser:
    def __init__(self):
        self._tasks = []

    def add_task(self, task) -> 'Parser':
        self._tasks.append(task)
        return self

    def parse(self, key: str, content: str) -> 'Parser':
        soup = BeautifulSoup(content, 'html.parser')
        for task in self._tasks:
            task.execute(key, soup)
        return self

    def parse_dict(self, content: dict) -> 'Parser':
        for k, v in content.items():
            self.parse(k, v)
        return self


if __name__ == '__main__':
    pass
    # page = '<div><span class="tm-article-datetime-published"><time datetime="2023-01-23T11:31:09.000Z" title="2023-01-23, ' \
    #        '14:31">23  янв   в 14:31</time></span></div>'
    # soup = BeautifulSoup(page, 'html.parser')
    # r = soup.find('span', class_='tm-article-datetime-published')
    # s = r.next.attrs['datetime']
    # print(s)
    # # print(datetime.strptime(s, '%Y/%m/%d %H:%M:%S.%f'))
    #
    # print(datetime.now())
    # x = datetime(2001, 1, 2, 10, 11, 12)
    # print(x)
