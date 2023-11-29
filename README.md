# Что сделал и где вопросы

Повторил модель с семинара, ее метрика MAPE@10 на валидационных данных с бота выше 0.063. 

Однако не закончил оборачивать модель в сервис из-за ошибки приложенной в скрине. 
Поэтому не протестил на потоке с бота. 
![Снимок экрана 2023-11-29 162907](https://github.com/Sergey-Kit/RecoServiceTemplate/assets/82327055/4997c3a7-02c1-4595-be25-ba1318f04230)

2) Реализовал два первых эксперимента в ноутбуке (/notebooks/itmo_recsys_dz_3_kNN.ipynb) 

3)Обертка готова. Необходимо убрать баги. 

# Шаблон сервиса рекомендаций

## Подготовка

### Python

В данном шаблоне используется Python3.8, однако вы можете использовать более свежие версии, если хотите. 
Но мы не гарантируем, что все будет работать.

### Make

[Make](https://www.gnu.org/software/make/) - это очень популярная утилита, 
предназначенная для преобразования одних файлов в другие через определенную последовательность команд. 
Однако ее можно использовать для исполнения произвольных последовательностей команд. 
Команды и правила их исполнения прописываются в `Makefile`.

Мы будем активно использовать `make` в данном проекте, поэтому рекомендуем познакомится с ней поближе. 

На MacOS и *nix системах `make` обычно идет в комплекте или ее можно легко установить. 
Некоторые варианты, как можно поставить `make` на `Windows`,
описаны [здесь](https://stackoverflow.com/questions/32127524/how-to-install-and-use-make-in-windows).

### Poetry

[Poetry](https://python-poetry.org/) - это удобный инструмент для работы с зависимостями в Python. 
Мы будем использовать его для подготовки окружения.

Поэтому перед началом работы необходимо выполнить [шаги по установке](https://python-poetry.org/docs/#installation).


## Виртуальное окружение

Мы будем работать в виртуальном окружении, которое создадим специально для данного проекта. 
Если вы не знакомы с концепцией виртуальных окружений в Python, обязательно 
[познакомьтесь](https://docs.python.org/3.8/tutorial/venv.html). 
Мы рекомендуем использовать отдельное виртуальное окружение для каждого вашего проекта.

### Инициализация окружения

Выполните команду
```
make setup
```

Будет создано новое виртуальное окружение в папке `.venv`.
В него будут установлены пакеты, перечисленные в файле `pyproject.toml`.

Обратите внимание: если вы один раз выполнили `make setup`, при попытке повторного ее выполнения ничего не произойдет, 
поскольку единственная ее зависимость - директория `.venv` - уже существует.
Если вам по какой-то причине нужно пересобрать окружение с нуля, 
выполните сначала команду `make clean` - она удалит старое окружение.

### Установка/удаление пакетов

Для установки новых пакетов используйте команду `poetry add`, для удаления - `poetry remove`. 
Мы не рекомендуем вручную редактировать секцию с зависимостями в `pyproject.toml`.

## Линтеры, тесты и автоформатирование

### Автоформатирование

Командой `make format` можно запустить автоматическое форматирование вашего кода.

Сейчас ее выполнение приведет лишь к запуску [isort](https://github.com/PyCQA/isort) - утилиты 
для сортировки импортов в нужном порядке. 
При желании вы также можете добавить другие инструменты, например [black](https://github.com/psf/black) или 
[yapf](https://github.com/google/yapf), которые могут действительно отформатировать код.


### Статическая проверка кода

Командой `make lint` вы запустите проверку линтерами - инструментами для статического анализа кода. 
Они помогают выявить ошибки в коде еще до его запуска, а также обнаруживают несоответствия стандарту 
[PEP8](https://peps.python.org/pep-0008).

### Тесты

Командой `make test` вы запустите тесты при помощи утилиты [pytest](https://pytest.org/). 


## Запуск приложения

### Способ 1: Python + Uvicorn

```
python main.py
```

Приложение запустится локально, в одном процессе. 
Хост и порт по умолчанию: `127.0.0.1` и `8080`.
Их можно изменить через переменные окружения `HOST` и `PORT`.

Управляет процессом легковесный [ASGI](https://asgi.readthedocs.io/en/latest/) server [uvicorn](https://www.uvicorn.org/).

Обратите внимание: для запуска нужно использовать `python` из окружения проекта.

### Способ 2: Uvicorn

```
uvicorn main:app
```

Очень похож на предыдущий, только запуск идет напрямую.
Хост и порт можно передать через аргументы командной строки.

Обратите внимание: для запуска нужно использовать `uvicorn` из окружения проекта.


### Способ 3: Gunicorn

```
gunicorn main:app -c gunicorn.config.py
```

Способ похож на предыдущий, только вместо `uvicorn` используется
более функциональный сервер [gunicorn](https://gunicorn.org/) (`uvicorn` используется внутри него).
Параметры задаются через конфиг, хост и порт можно задать 
через переменные окружения или аргументы командной строки.

Сервис запускается в несколько параллельных процессов, по умолчанию их число
равно числу ядер процессора.

Обратите внимание: для запуска нужно использовать `gunicorn` из окружения проекта.

### Способ 4: Docker

Делаем все то же самое, но внутри docker-контейнера. 
Если вы не знакомы с [docker](https://www.docker.com/), обязательно познакомьтесь.

Внутри контейнера можно использовать любой из способов, описанных выше.
В продакшене рекомендуется использовать `gunicorn`.

Собрать и запустить образ можно командой

```
make run
```

## CI/CD

Когда вы делаете пуш в гит (в любую ветку), выполняется процесс CI. 
Что именно выполняется в этом процессе и как он триггерится (в данном случае по пушу), 
описывается в специальных `.yaml` конфигах в папке `.github/workflows`.

Сейчас там есть только один конфиг, который запускает процесс, в котором создается виртуальное окружение,
прогоняются линтеры и тесты. Если что-то пошло не так, процесс падает с ошибкой и в Github появляется красный крестик.
Вам нужно посмотреть логи, исправить ошибку и запушить изменения.
