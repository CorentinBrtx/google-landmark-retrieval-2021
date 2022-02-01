import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


import glob
import logging
import os
from argparse import ArgumentParser
from typing import Optional

import pandas as pd
from src.constants import Constants
from src.data.utils import get_path, int_to_string


def transfer_images(data_dir: str, clean_csv: str, nb_landmarks: int = -1):
    """
    Transfer the images from the temporary folder to the train folder by keeping only the landmarks in the clean_csv file.

    Parameters
    ----------
    data_dir : str
        Directory where the data is stored.
    clean_csv : str
        Clean csv file containing the landmarks.
    nb_landmarks : int, optional
        Number of landmarks to keep, -1 to keep all, by default -1
    """
    clean_data = pd.read_csv(clean_csv)
    images_clean = {}
    for _, row in clean_data.iterrows():
        for image in row["images"].split(" "):
            images_clean[image] = row["landmark_id"]

    landmarks = sorted(list(set(images_clean.values())))
    max_landmark = landmarks[nb_landmarks]

    train_csv = os.path.join(data_dir, "train.csv")

    if os.path.exists(train_csv):
        current_data = pd.read_csv(train_csv).set_index("image_id")["landmark_id"].to_dict()
    else:
        current_data = {}

    images_paths = glob.glob(os.path.join(data_dir, "images_temp", "*/*/*/*.jpg"))
    for image in images_paths:
        image_name = os.path.splitext(os.path.basename(image))[0]
        if image_name not in images_clean or images_clean[image_name] > max_landmark:
            os.system("rm " + image)
        elif image_name not in current_data:
            os.system(
                f"mkdir -p {os.path.dirname(get_path(os.path.join(data_dir, 'train'), image_name))}; "
                + f"mv {image} $_"
            )
            current_data[image_name] = images_clean[image_name]

    new_data = pd.DataFrame.from_dict(current_data, orient="index", columns=["landmark_id"])
    new_data["image_id"] = new_data.index
    new_data[["image_id", "landmark_id"]].to_csv(train_csv, index=False)


def download_and_sort(
    data_dir: str,
    clean_csv: Optional[str] = None,
    begin: int = 0,
    end: int = 0,
    nb_landmarks: int = -1,
):
    """
    Download the images from the Google Landmark dataset train set, and keep only the ones corresponding to the landmarks in the clean_csv file.

    Parameters
    ----------
    data_dir : str
        Directory where the data will be downloaded.
    clean_csv : Optional[str], optional
        Path to the clean_csv file, if None, the file will be downloaded, by default None
    begin : int, optional
        first file to be downloaded, between 0 and 500, by default 0
    end : int, optional
        last file to be downloaded, between begin and 500, by default 0
    nb_landmarks : int, optional
        number of landmarks to keep from the clean_csv file, -1 to keep all landmarks, by default -1
    """

    data_dir = os.path.abspath(data_dir)

    os.makedirs(os.path.join(data_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "images_temp"), exist_ok=True)

    if clean_csv is None:
        clean_csv = os.path.join(data_dir, "train_clean.csv")

    if not os.path.exists(clean_csv):
        os.system("curl -Os https://s3.amazonaws.com/google-landmark/metadata/train_clean.csv")
        os.system("mv ./train_clean.csv " + clean_csv)

    logging.info("Downloading images from Google Landmark Dataset")

    os.system(
        " ".join(
            [
                f"bash {os.path.join(Constants.SRC_FOLDER, 'data','google_download.sh')}",
                "train",
                begin,
                end,
                data_dir,
                os.path.join(data_dir, "images_temp"),
            ]
        )
    )

    logging.info("Transfering downloaded images")

    transfer_images(data_dir=data_dir, clean_csv=clean_csv, nb_landmarks=nb_landmarks)

    for i in range(begin, end + 1):
        os.system(f"rm {os.path.join(data_dir, f'md5.images_{int_to_string(i)}.txt')}")

    os.system(f"rm -r {os.path.join(data_dir,'images_temp')}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Download the training database and sort to keep only clean samples"
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        dest="data_dir",
        help="Directory to contain the data.",
        required=True,
    )
    parser.add_argument(
        "-b",
        "--begin",
        dest="begin",
        type=int,
        help="Begin index of the images to download (included)",
        default=0,
    )
    parser.add_argument(
        "-e",
        "--end",
        dest="end",
        type=int,
        help="End index of the images to download (included)",
        default=0,
    )
    parser.add_argument(
        "-n",
        "--nb_landmarks",
        dest="nb_landmarks",
        type=int,
        help="Number of landmarks to keep (-1 to keep all)",
        default=-1,
    )
    parser.add_argument(
        "-c", "--clean_csv", dest="clean_csv", help="Path to the clean csv file", default=None
    )

    args = parser.parse_args()

    os.environ["LANDMARK_RETRIEVAL_DATA_DIR"] = args.data_dir

    for k in range(args.begin, args.end + 1, 6):
        download_and_sort(args.data_dir, args.clean_csv, k, min(k + 5, args.end), args.nb_landmarks)
