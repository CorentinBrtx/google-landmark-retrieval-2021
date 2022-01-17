import glob
import logging
import os

import pandas as pd

from .utils import get_path, int_to_string


def transfer_images(clean_csv: str, nb_landmarks: int = -1):
    clean_data = pd.read_csv(clean_csv)
    images_clean = {}
    for _, row in clean_data.iterrows():
        for image in row["images"].split(" "):
            images_clean[image] = row["landmark_id"]

    landmarks = sorted(list(images_clean.values()))
    max_landmark = landmarks[nb_landmarks]

    if os.path.exists("./data/train.csv"):
        current_data = (
            pd.read_csv("./data/train.csv").set_index("image_id")["landmark_id"].to_dict()
        )
    else:
        current_data = {}

    images_paths = glob.glob(os.path.join("./data/images_temp", "*/*/*/*.jpg"))
    for image in images_paths:
        image_name = os.path.splitext(os.path.basename(image))[0]
        if image_name not in images_clean or images_clean[image_name] > max_landmark:
            os.system("rm " + image)
        elif image_name not in current_data:
            os.system(
                f"mkdir -p {os.path.dirname(get_path('./data/train', image_name))}; mv {image} $_"
            )
            current_data[image_name] = images_clean[image_name]

    new_data = pd.DataFrame.from_dict(current_data, orient="index", columns=["landmark_id"])
    new_data["image_id"] = new_data.index
    new_data[["image_id", "landmark_id"]].to_csv("./data/train.csv", index=False)


def download_and_sort(
    clean_csv: str = "./data/train_clean.csv", begin: int = 0, end: int = 10, nb_landmarks: int = -1
):

    os.makedirs("./data/train", exist_ok=True)
    os.makedirs("./data/images_temp", exist_ok=True)

    if not os.path.exists(clean_csv):
        os.system("curl -Os https://s3.amazonaws.com/google-landmark/metadata/train_clean.csv")
        os.system("mv ./train_clean.csv " + clean_csv)

    logging.info("Downloading images from Google Landmark Dataset")

    os.system(
        "bash ./src/data/download.sh train " + str(begin) + " " + str(end) + " ./data/images_temp"
    )

    logging.info("Transfering downloaded images")

    transfer_images(clean_csv, nb_landmarks)

    for i in range(begin, end + 1):
        os.system(f"rm images_{int_to_string(i)}.tar")
        os.system(f"rm md5.images_{int_to_string(i)}.txt")

    os.system("rm -r ./data/images_temp")
