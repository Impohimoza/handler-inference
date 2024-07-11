import cv2
from inference import YOLOv5Handler


def main(image_path: str):
    img = cv2.imread(r'{}'.format(image_path))
    if img is None:
        print(f"Unable to open image file {image_path}")
        return

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    handler = YOLOv5Handler()
    handler.on_start()

    results = handler.handle(img)
    results.render()
    for img in results.ims:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow('Inference', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    handler.on_exit()


if __name__ == "__main__":
    while (image_path := input("Введите путь к изображению или напишите q - ")) != "q":
        main(image_path)
