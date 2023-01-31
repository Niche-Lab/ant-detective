# native imports
import os
import pandas as pd

# pytorch imports
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image


class Find_Ant_Dataset(Dataset):
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
        # get label
        label = self.img_labels.iloc[idx, 1]
        # data transformation
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def create_loader(dir_data: str) -> dict(str, DataLoader):
    loader = dict({})
    for setname in ["train", "test", "val"]:
        dataset = Find_Ant_Dataset(os.path.join(dir_data, setname))
        is_test = setname == "test"  # when is test, we don't shuffle
        loader[setname] = DataLoader(
            dataset, batch_size=16, shuffle=~is_test, num_workers=4
        )
    return loader
