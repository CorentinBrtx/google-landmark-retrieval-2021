# Kaggle Challenge: Google Landmark Retrieval 2021

This is part of a project for the Deep Learning course of CentraleSupélec.

This repository contains code for the Kaggle challenge: [Google Landmark Retrieval 2021](https://www.kaggle.com/c/google-landmark-retrieval-2021).

The idea of the model used comes from the 2020 winner [google_landmark_retrieval_1st_place_solution](https://github.com/seungkee/google_landmark_retrieval_2020_1st_place_solution), and the code was adapted from the adaptive margin loss tutorial [Emkademy](https://github.com/kivancyuksel/emkademy).

## Installation

To install the required packages, run the following command:

    pip install -r requirements.txt

## Usage

### Data download

To download the data, you can use the following command:

    python ./src/scripts/download.py --data_dir path/to/data/directory

For more information, use:

    python ./src/scripts/download.py --help

### Model training

To train your own model, you can use the following command:

    python ./src/scripts/train.py model_name --data_dir path/to/data/directory

For more information, use:

    python ./src/scripts/train.py --help

### Embeddings extraction

To use a model to extract embeddings from a dataset, you can use the following command:

    python ./src/scripts/test.py path/to/model --data_dir path/to/test/images

For more information, use:

    python ./src/scripts/test.py --help

## Kaggle submission

To actually submit your model to the Kaggle competition, you can use the following command, you can use the provided notebook `kaggle_main.ipynb`, by modifying it to specify the relevant parameters.
