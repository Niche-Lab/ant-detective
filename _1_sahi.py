from paths import PathFinder
from pathlib import Path
import sys
PATHS = PathFinder()
# insert to the first position
sys.path.insert(0, str(PATHS["DIR_PYNICHE"]))

from pyniche.data.yolo.API import YOLO_API

DIR_DATA = PATHS["DIR_DATA"]
yolo = YOLO_API(DIR_DATA / "study1")


from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict


