from ultralytics import YOLO
from dotenv import load_dotenv
import os
os.getcwd()

load_dotenv(".env")
DIR_DATA = os.getenv("DIR_DATA") + "/paper"
model = YOLO("yolo8n_study2.pt")

dir_4x10 = os.path.join(DIR_DATA, "1x1")
model.predict(dir_4x10, device="mps", 
              save=True,
              show_conf=False,
              line_width=1,
              show_labels=False,
              conf=0.5,
              save_txt=True,)