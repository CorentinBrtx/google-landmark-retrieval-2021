import os


def int_to_string(n):
    if n < 10:
        return "00" + str(n)
    elif n < 100:
        return "0" + str(n)
    return n


def get_path(folder, image_id):
    return os.path.join(folder, f"{image_id[0]}/{image_id[1]}/{image_id[2]}/{image_id}.jpg")


def get_id(image_path):
    return os.path.splitext(os.path.basename(image_path))[0]
