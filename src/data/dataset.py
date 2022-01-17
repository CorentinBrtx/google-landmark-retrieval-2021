import cv2 as cv
import pandas as pd
from torch.utils.data import Dataset
from utils import get_path


class GoogleLandmarkDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        landmarks = sorted(list(set(self.img_labels["landmark_id"])))
        self.landmark_id_to_label = {landmarks[i]: i for i in range(len(landmarks))}
        self.img_dir = img_dir
        self.num_classes = len(landmarks)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = get_path(self.img_dir, self.img_labels.iloc[idx, 0])
        image = cv.imread(img_path)
        label = self.landmark_id_to_label[self.img_labels.iloc[idx, 1]]  # pylint: disable=no-member
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
