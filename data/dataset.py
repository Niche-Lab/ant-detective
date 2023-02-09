# native imports
import os
import pandas as pd

# pytorch imports
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image


class PepSuc_Dataset(Dataset):
    """
    File structure:
    data/
        train/
            1.jpg
            2.jpg
            ...
            annotation.txt
        test/
            1.jpg
            2.jpg
            ...
            annotation.txt

    annotation.txt:
    # filename label
    1.jpg 3
    2.jpg 5
    3.jpg 0
    ...
    """

    def __init__(self, dir_img, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(
            os.path.join(dir_img, "annotation.txt"), header=None
        )
        self.dir_img = dir_img
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # get image matrix
        img_path = os.path.join(self.dir_img, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        # get label: sucrose and peptone
        label = self.img_labels.loc[idx, ["sucrose", "peptone"]].values
        # data transformation
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def create_loader(name_data: str) -> dict(str, DataLoader):
    # get root directory
    ROOT = os.path.dirname(os.path.abspath(__file__))
    # create dataloader
    loader = dict({})
    for setname in ["train", "test", "val"]:
        is_test = setname == "test"  # when is test, we don't shuffle
        # init dataset
        if name_data == "peptone_sucrose":
            dataset = PepSuc_Dataset(os.path.join(ROOT, "peptone_sucrose", setname))
        # init dataloader
        loader[setname] = DataLoader(
            dataset, batch_size=16, shuffle=~is_test, num_workers=4
        )
    return loader
