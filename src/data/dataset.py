import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
from src.data.utils import get_path


class GoogleLandmarkDataset(Dataset):
    def __init__(
        self, annotations_file, img_dir, transform=None, target_transform=None, load_all=False
    ):
        self.img_labels = pd.read_csv(annotations_file)
        landmarks = sorted(list(set(self.img_labels["landmark_id"])))
        self.landmark_id_to_label = {landmarks[i]: i for i in range(len(landmarks))}
        self.img_dir = img_dir
        self.num_classes = len(landmarks)
        self.transform = transform
        self.target_transform = target_transform
        self.load_all = load_all

        if self.load_all:
            self.images = []
            self.labels = []
            for _, row in self.img_labels.iterrows():
                image, label = self.get_image_and_label(row["image_id"], row["landmark_id"])
                self.images.append(image)
                self.labels.append(label)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        if self.load_all:
            return self.images[idx], self.labels[idx]
        else:
            img_path = get_path(self.img_dir, self.img_labels.iloc[idx, 0])
            landmark_id = self.img_labels.iloc[idx, 1]  # pylint: disable=no-member
            return self.get_image_and_label(img_path, landmark_id)

    def get_image_and_label(self, img_path: str, landmark_id: str):
        image = read_image(img_path)
        label = self.landmark_id_to_label[landmark_id]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
