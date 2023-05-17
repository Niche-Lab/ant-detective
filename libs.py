from PIL import Image
import numpy as np
import cv2


def split_img(image, rate=8, newsize=(640, 640)):
    """
    split image into rate x rate pieces
    ---
    params:
        image: np.ndarray or str
        rate: int
        newsize: tuple
    return:
        list of PIL.Image
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif isinstance(image, str):
        image = Image.open(image)
    W, H = image.size
    sw = int(W / rate)
    sh = int(H / rate)
    imgs = []
    for i in range(rate):
        for j in range(rate):
            img_cropped = image.crop((i * sw, j * sh, (i + 1) * sw, (j + 1) * sh))
            img_resized = img_cropped.resize(newsize)
            imgs.append(np.array(img_resized)[:, :, ::-1])
    return imgs


def compute_area(xyxy):
    x1, y1, x2, y2 = xyxy
    return (x2 - x1) * (y2 - y1)


def get_quantile_area(results, quantile=0.8):
    """
    get quantile area of all bounding boxes
    ---
    params:
        results: output from YOLO model.predict()
    return:
        float
    """
    areas = []
    for result in results:
        for box in np.array(result.boxes.xyxy.cpu()):
            areas.append(compute_area(box))
    thred = np.quantile(areas, quantile)
    return thred
