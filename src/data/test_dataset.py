import glob
import os

from src.data.utils import get_id
from torch.utils.data import Dataset
from torchvision.io import read_image


class TestLandmarkDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_paths = glob.glob(os.path.join(img_dir, "*/*/*/*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        image = read_image(img_path).float()
        if self.transform:
            image = self.transform(image)
        return image, get_id(img_path)
