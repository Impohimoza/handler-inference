from abc import ABC, abstractmethod
from typing import Any
import torch


class Handler(ABC):
    """Интерфейс компонента аналитики."""
    def on_start(self, *args, **kwargs) -> Any:
        """Метод, который вызывается перед началом работы компонента.
        В него стоит помещать инициализацию:
        - необходимых ресурсов, связанных с хранением состояния;
        - весов моделей;
        - коллекций;
        - объектов.
        Общая рекомендация следующая: в конструкторе оставляем инициализацию
        только простых типов (int, float, str).
        Идея в том, чтобы сократить объём копирования при запуске хендлера в
        отдельном fork-процессе.
        По умолчанию ничего не делает.
        """
        pass

    @abstractmethod
    def handle(self, *args, **kwargs) -> Any:
        """Метод, выполняющий основную работу."""
        raise NotImplementedError

    def on_exit(self, *args, **kwargs) -> None:
        """Метод, вызывающийся после завершения работы компонента.
        Должен освобождать инициализированные ресурсы.
        По умолчанию ничего не делает.
        """
        pass


class YOLOv5Handler(Handler):
    """Обработчик для инференса модели детекции YOLOv5"""
    def __init__(self, model_name: str = 'yolov5s'):
        self.model_name = model_name
        self.model = None

    def on_start(self, *args, **kwargs) -> Any:
        """Метод для инициализации компонента"""
        self.model = torch.hub.load('ultralytics/yolov5',
                                    self.model_name,
                                    pretrained=True
                                    )

    def handle(self, img) -> Any:
        """Метод для обработки инференса модели детекции"""
        if self.model is None:
            raise Exception("Компонент не инициализирован")
        results = self.model(img)

        # Выбираем только тарелки
        filtered_results = results.pred[0][results.pred[0][:, -1] == 45]
        results.pred[0] = filtered_results
        return results

    def on_exit(self, *args, **kwargs) -> None:
        """Метод для освобождения инициализированных ресурсов"""
        del self.model
