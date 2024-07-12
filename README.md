# handler-inference

## Описание

Этот проект представляет собой обёртку для инференса модели детекции YOLOv5, написанную на Python.

## Установка

1. Клонируйте репозиторий:
    ```
    git clone https://github.com/Impohimoza/handler-inference.git
    cd handler-inference
    ```

2. Создайте и активируйте виртуальное окружение:
    ```
    python -m venv venv
    source venv/bin/activate
    pip3 install -U pip
    ```

3. Установите зависимости:
    ```sh
    pip install -r requirements.txt
    ```

## Использование


1. Запустите `main.py`:
    ```sh
    python main.py path/to/your/image.jpg --confidence 0.5
    # Если не указать --confidence, то значение будет 0.5
    ```

2. Следуйте инструкциям в консоли.
