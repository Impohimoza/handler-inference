from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Tuple, List

import numpy as np
import torch


@dataclass
class Detection:
    absolute_box: Tuple[int, int, int, int]
    relative_box: Tuple[float, float, float, float]
    score: float
    label_as_int: int
    label_as_str: str


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
    def __init__(self, model_name: str = 'yolov5s', confidence: float = 0.5):
        self.model_name: str = model_name
        self.confidence: float = confidence
        self.model: Any = None

    def on_start(self, *args, **kwargs) -> None:
        """Метод для инициализации компонента"""
        self.model = torch.hub.load(
            'ultralytics/yolov5',
            self.model_name,
            pretrained=True
        )

    def handle(self, img: np.ndarray) -> List[Detection]:
        """Метод для обработки инференса модели детекции"""
        if self.model is None:
            raise Exception("The component is not initialized")
        results: Any = self.model(img)

        # Выбираем только тарелки
        filtered_results: Any = results.pred[0][results.pred[0][:, -1] == 45]

        detections: List[Detection] = []
        for det in filtered_results:
            x1, y1, x2, y2, conf, clss = det
            score: float = conf.item()
            if score < self.confidence:
                continue
            absolute_box: Tuple[int, int, int, int] = (
                int(x1),
                int(y1),
                int(x2),
                int(y2)
            )
            relative_box: Tuple[float, float, float, float] = (
                x1.item() / img.shape[1],
                y1.item() / img.shape[0],
                x2.item() / img.shape[1],
                y2.item() / img.shape[0]
            )
            label_as_int: int = int(clss.item())
            label_as_str: str = self.model.names[label_as_int]

            detections.append(Detection(
                absolute_box,
                relative_box,
                score,
                label_as_int,
                label_as_str
            ))

        return detections

    def on_exit(self, *args, **kwargs) -> None:
        """Метод для освобождения инициализированных ресурсов"""
        del self.model
