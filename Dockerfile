# Используем официальный базовый образ Python
FROM python:3.10-slim

# Устанавливаем переменные окружения
ENV DATA_ROOT /data
ENV PROJECT_ROOT /app

# Создаем директории
RUN mkdir -p $DATA_ROOT
RUN mkdir -p $PROJECT_ROOT

# Устанавливаем зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Копируем только файлы, необходимые для установки зависимостей, чтобы избежать ненужных сборок
COPY requirements.txt $PROJECT_ROOT/

# Переходим в рабочую директорию
WORKDIR $PROJECT_ROOT

# Устанавливаем зависимости Python
RUN pip install --no-cache-dir -r requirements.txt

# Копируем остальные файлы проекта
COPY . .

# Даем права на выполнение скриптов
RUN chmod +x data/get_data.sh

# Запускаем скрипт для получения данных
RUN ./data/get_data.sh

# Задаем команду по умолчанию
CMD ["python", "lib/run.py"]
