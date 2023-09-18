"""
This script process the original images to:
    (1) original: resize to 512x512
    (2) 2x2: crop the original imagei to 4 images of 512 x 512

The image naming convention is:
    t<trial>_<image_id>_0<count>.jpg
    e.g., t1_1_05.jpg is the image of trial 1, image id 1, and contains 5 ants
"""
import os
import sys
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(__file__))
SIZE_TARGET = (512, 512)
DIR_DATA = os.path.join(ROOT, "data")
DIR_IMAGE = os.path.join(DIR_DATA, "raw", "peptone_sucrose")
DIR_OUT_ORIGINAL = os.path.join(DIR_DATA, "ps_original")
DIR_OUT_GRID = os.path.join(DIR_DATA, "ps_grid")


def main():
    create_dir(DIR_OUT_ORIGINAL)
    create_dir(DIR_OUT_GRID)
    for t in range(9):  # 9 trials
        trial_name = "Trial %d" % (t + 1)
        trial_dir = os.path.join(DIR_IMAGE, trial_name)
        ls_imgs = os.listdir(trial_dir)
        for idx, img_name in enumerate(ls_imgs):
            # read image
            path_img = os.path.join(trial_dir, img_name)
            img = Image.open(path_img)
            # process original and grid images
            img_original = img.resize(SIZE_TARGET)
            img_grid = grid_image(img, SIZE_TARGET)
            # save original image
            path_out_original = os.path.join(
                DIR_OUT_ORIGINAL, "t%d_%d_0.jpg" % (t + 1, idx + 1)
            )
            img_original.save(path_out_original)
            # save grid images
            for key in img_grid:
                if key == "top-left":
                    suffix = "tl"
                elif key == "top-right":
                    suffix = "tr"
                elif key == "bottom-left":
                    suffix = "bl"
                elif key == "bottom-right":
                    suffix = "br"
                path_out_grid = os.path.join(
                    DIR_OUT_GRID, "t%d_%d_%s_0.jpg" % (t + 1, idx + 1, suffix)
                )
                img_grid[key].save(path_out_grid)


def create_dir(dir_out):
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)


def grid_image(image: Image, size_target=SIZE_TARGET):
    """
    return top-left, top-right, bottom-left, bottom-right images
    """
    W, H = image.size
    images = dict(
        {
            "top-left": image.crop((0, 0, int(W / 2), int(H / 2))),
            "top-right": image.crop((int(W / 2), 0, W, int(H / 2))),
            "bottom-left": image.crop((0, int(H / 2), int(W / 2), H)),
            "bottom-right": image.crop((int(W / 2), int(H / 2), W, H)),
        }
    )
    # resize
    for key in images:
        images[key] = images[key].resize(size_target)
    # return
    return images


if __name__ == "__main__":
    main()
