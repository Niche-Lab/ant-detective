from PIL import Image
import numpy as np
import os

# dl imports
import supervision as sv
from ultralytics import YOLO

# local imports
from models.sam import seg_by_sam
from libs import compute_area, tile_img, get_quantile_area

ROOT = os.path.abspath(os.path.dirname(__file__))


class Niche_Batch:
    def __init__(self, path_img, rate=8, newsize=(640, 640)):
        # images
        self.image = Image.open(path_img)
        self.ls_tiles = []  # list of tiled PIL.Image from self.image
        self.ls_bbox = []  # list of bbox ant image from YOLO model
        # config
        self.rate = rate  # split image into rate x rate pieces
        self.newsize = newsize  # resize each piece to newsize
        # results
        self.dets = []  # list of Niche_CV_Object

    def tile_images(self, show=False):
        self.ls_tiles = tile_img(self.image, self.rate, self.newsize)
        if show:
            self.plot_tiles_grid()

    def yolo_detect(self, path_model, device="cpu", thred=0.8):
        # detect via YOLO model
        model = YOLO(path_model)
        results = model.predict(self.ls_tiles, device=device)
        # set filtering threshold
        thred = get_quantile_area(results, thred)
        print(f"threshold: {thred}")
        # load results to self.dets
        for i, result in enumerate(results):
            xyxy = preprocess_xyxy(result.boxes.xyxy.cpu(), thred)
            detection = Niche_CV_Object(self.ls_tiles[i], xyxy)
            self.dets.append(detection)

    def set_bbox_images(self, show=False):
        for det in self.dets:
            if len(det) > 0:
                # use extend instead of append to flatten the list
                self.ls_bbox.extend(det.get_bbox_image(resize=self.newsize))
        if show:
            self.plot_bbox_grid()

    def plot_image(self, index):
        self.dets[index].plot_image()

    def plot_tiles_grid(self):
        sv.plot_images_grid(self.ls_tiles, grid_size=(self.rate, self.rate))

    def plot_detect_grid(self):
        sv.plot_images_grid(
            [det.get_annotated_image() for det in self.dets],
            grid_size=(self.rate, self.rate),
        )

    def plot_bbox_grid(self, n=10):
        sv.plot_images_grid(
            [np.array(i) for i in self.ls_bbox[: n**2]],
            grid_size=(n, n),
        )

    def export_yolo(self, split=0.9):
        dir_out, path_yaml = handle_yolo_folders(ROOT)
        # create yaml file for yolov5
        with open(path_yaml, "w") as f:
            f.write(
                f"""
                    train: {os.path.join(dir_out, "train", "images")}
                    val: {os.path.join(dir_out, "val", "images")}
                    nc: 1
                    names: ['ant']
                """
            )
        # split train and val
        n = len(self.ls_bbox)
        n_train = int(n * split)
        # create train and val images and labels
        self._export_yolo(self.ls_bbox[:n_train], os.path.join(dir_out, "train"))
        self._export_yolo(self.ls_bbox[n_train:], os.path.join(dir_out, "val"))

    def _export_yolo(self, ls_bbox, dir_out):
        dir_imgs = os.path.join(dir_out, "images")
        dir_labels = os.path.join(dir_out, "labels")
        for i, img in enumerate(ls_bbox):
            # save image
            img.save(os.path.join(dir_imgs, f"img_{i}.jpg"))
            # save label
            with open(os.path.join(dir_labels, f"img_{i}.txt"), "w") as f:
                f.write(f"0 0.5 0.5 1 1\n")


class Niche_CV_Object(sv.Detections):
    def __init__(self, image, xyxy):
        super().__init__(xyxy)
        self.image = image
        self.mask = None

    def set_mask_by_sam(self, sam_predictor):
        self.mask = seg_by_sam(sam_predictor, self.image, self.xyxy)

    def plot_image(self, figsize=(8, 8)):
        annt_img = self.get_annotated_image()
        sv.plot_image(annt_img, figsize)

    def get_annotated_image(self):
        annotated_image = self.image.copy()
        # annotate mask
        if self.mask is not None:
            annotator = sv.MaskAnnotator()
            annotated_image = annotator.annotate(annotated_image, self)
        # annotate box
        annotator = sv.BoxAnnotator()
        annotated_image = annotator.annotate(annotated_image, self)
        return annotated_image

    def get_bbox_image(self, resize=None):
        ls_imgs = []
        for xyxy in self.xyxy:
            x1, y1, x2, y2 = xyxy
            img_crop = Image.fromarray(self.image[y1:y2, x1:x2])
            if resize is not None:
                img_crop = img_crop.resize(resize)
            ls_imgs.append(img_crop)
        return ls_imgs


def preprocess_xyxy(xyxy, thred=1e9):
    xyxy = np.array(xyxy, dtype=int).reshape(-1, 4)
    # filter by bbox area
    areas = []
    for box in xyxy:
        areas.append(compute_area(box))
    xyxy = xyxy[np.array(areas) < thred]
    return xyxy


def handle_yolo_folders(root):
    # export YOLO
    DIR_OUT = os.path.join(root, "data", "yolo_ant")
    if not os.path.exists(DIR_OUT):
        os.mkdir(DIR_OUT)
    PATH_YAML = os.path.join(DIR_OUT, "data.yaml")
    for split in ["train", "val"]:
        for folder in ["images", "labels"]:
            path = os.path.join(DIR_OUT, split, folder)
            if not os.path.exists(path):
                os.mkdir(path)
    # return
    return DIR_OUT, PATH_YAML
