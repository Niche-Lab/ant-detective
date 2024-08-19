import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def plot_split_imgs(ls_imgs, nrows, results=None):
    """
    plot split images and their results if provided
    ---
    params:
        ls_imgs: list of PIL.Image
        nrows: int, how many rows (and columns) to plot
        results: predictions from YOLO model.predict()

    """
    fig, axes = plt.subplots(nrows, nrows, figsize=(nrows * 5, nrows * 5))
    plt.tight_layout()

    for i in range(nrows):
        for j in range(nrows):
            axes[j, i].imshow(ls_imgs[i * nrows + j])
            axes[j, i].axis("off")
            if results is not None:
                for result in results[i * nrows + j]:
                    # send to cpu
                    for box in np.array(result.boxes.xyxy.cpu()):
                        x1, y1, x2, y2 = box
                        w, h = x2 - x1, y2 - y1
                        axes[j, i].add_patch(
                            Rectangle(
                                (x1, y1),
                                w,
                                h,
                                linewidth=1,
                                edgecolor="r",
                                facecolor="r",
                                alpha=0.2,
                            )
                        )
    plt.show()
