from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
file = "/Users/niche/Pictures/cow.jpeg"

img = Image.open(file)
model = YOLO("yolov8n.pt")
out = model(img)

ls_cls = out[0].names
ls_det_cls = out[0].boxes.cls.numpy()
ls_det_cls = [ls_cls[i] for i in ls_det_cls]
ls_det_xyxy = out[0].boxes.xyxy.numpy()
import matplotlib.pyplot as plt


# Open an image


img = Image.open(file)

# Define the coordinates of the bounding box (top-left and bottom-right corners)
# Format: (x1, y1, x2, y2)

# Draw the rectangle (bounding box)
draw = ImageDraw.Draw(img)
for xyxy, name in zip(ls_det_xyxy, ls_det_cls):
    print(name)
    draw.rectangle(xyxy, outline="red", width=3)
    font = ImageFont.load_default(size=100)
    # text_width, text_height = draw.textsize(name)
    text_x = xyxy[0]
    text_y = xyxy[3]
    draw.text((text_x, text_y), name, fill="red", font=font)
img
npimg = np.array(img)
import av
av.VideoFrame.from_ndarray(npimg, format="bgr24")

# Optionally, save or show the image with the bounding box
image.save('annotated_image.jpg')
image.show()
plt.imshow(npimg)

# annotate bounding boxes on numpy array
