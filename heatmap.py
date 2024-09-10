from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
import tqdm

def vis_heatmap(ls_txt):
    heatmap = make_heatmap(ls_txt)
    plot_heatmap(heatmap)

def make_heatmap(ls_txt):
    ls_det = from_txt_to_det(ls_txt)
    ls_cir = from_det_to_cir(ls_det)
    heatmap = cir_to_heatmap(ls_cir)
    return heatmap

def from_txt_to_det(ls_txt):
    ls_det = []
    for f in tqdm.tqdm(ls_txt, desc="Reading txt files"):
        with open(f) as f:
            ls_det += [l.strip() for l in f.readlines()] 
    return ls_det

def from_det_to_cir(ls_det):
    ls_cir = []
    for det in tqdm.tqdm(ls_det, desc="Parsing detections"):
        # split by space
        dets = det.split(" ")
        x = float(dets[1])
        y = float(dets[2])
        w = float(dets[3])
        h = float(dets[4])
        r = (w * h) ** 0.5 / 2 # radius
        ls_cir += [(x, y, r)]
    return ls_cir

def cir_to_heatmap(ls_cir):
    # Image size
    image_size = (1000, 1000)
    # Create an empty image grid
    heatmap = np.zeros(image_size)
    # Resize the coordinates and plot circles
    for x, y, r in tqdm.tqdm(ls_cir, desc="Creating heatmap"):
        x_resized = int(x * image_size[0])
        y_resized = int(y * image_size[1])
        radius_resized = int(r * image_size[0])
        if radius_resized == 0:
            radius_resized = 1

        # Create a grid of the same size as the image
        Y, X = np.ogrid[:image_size[0], :image_size[1]]
        distance = np.sqrt((X - x_resized) ** 2 + (Y - y_resized) ** 2)

        # Add intensity to the heatmap
        num = -(distance**2)
        den = 2 * (radius_resized**2)
        heatmap += np.exp(num / den)
    heatmap = gaussian_filter(heatmap, sigma=5)
    # return
    return heatmap

def plot_heatmap(heatmap, max_value=6):
    plt.imshow(heatmap, cmap='hot', extent=(0, 1, 0, 1))
    plt.clim(0, max_value) # standardize the colorbar and c max
    plt.colorbar(label='Intensity')
    plt.title('Heatmap of Circle Distribution')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.axis("off")
    plt.show()