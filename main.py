import argparse
from typing import Any, List

import cv2
import numpy as np

from inference import YOLOv5Handler, Detection


def draw_detection(img: np.ndarray, detections: List[Detection]) -> np.ndarray:
    for detection in detections:
        x1, y1, x2, y2 = detection.absolute_box
        label: str = detection.label_as_str
        score: float = detection.score

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f'{label} {score:.2%}',
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    return img


def main(image_path: str, confidence: float = 0.5):
    img: np.ndarray = cv2.imread(r'{}'.format(image_path))
    if img is None:
        print(f"Unable to open image file {image_path}")
        return

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    handler: Any = YOLOv5Handler(confidence=confidence)
    handler.on_start()

    detections: List[Detection] = handler.handle(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = draw_detection(img, detections)
    cv2.imshow('Inference', img)
    # Логика показа изображения 10 секунд или до нажатия 'q'
    start_time = cv2.getTickCount()
    while (cv2.getTickCount() - start_time) / cv2.getTickFrequency() < 10:
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cv2.destroyAllWindows()

    handler.on_exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLOv5 Inference Script')
    parser.add_argument('image_path', type=str, help='Path to the image')
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='Confidence threshold for detection'
    )

    args = parser.parse_args()
    main(args.image_path, args.confidence)
