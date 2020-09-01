import argparse
import math
import os
import random
import xml.etree.ElementTree as ETree
from multiprocessing import Process

import yaml
from PIL import Image

ORIGIN_IMAGE_FOLDER = "image"
YOLO5_IMAGE_FOLDER = "images"

ORIGIN_LABEL_FOLDER = "image_annotation"
YOLO5_LABEL_FOLDER = "labels"

CLASSES = ["armor", "base", "watcher", "car"]

SRC_LIST = [
    "robomaster_Central China Regional Competition",
    "robomaster_Final Tournament",
    "robomaster_North China Regional Competition",
    "robomaster_South China Regional Competition",
]

DES_LIST = [
    "central",
    "final",
    "north",
    "south",
]


def split_list(src_list, split_ratio):
    split_index1 = math.floor(len(src_list) * split_ratio[0] / sum(split_ratio))
    split_index2 = math.floor(len(src_list) * sum(split_ratio[:2]) / sum(split_ratio))

    train = src_list[:split_index1]
    test = src_list[split_index1:split_index2]
    val = src_list[split_index2:]

    return train, test, val


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rearrange DJI ROCO dataset to VOC style.",
        epilog="Example: python3 roco2x.py --dji-roco-dir ~/DJI ROCO/",
    )

    parser.add_argument(
        "--dji-roco-dir",
        type=str,
        default="~/DJI ROCO/",
        help="dataset directory on disk",
    )

    parser.add_argument(
        "--split-ratio",
        nargs="+",
        type=int,
        default=[7, 2, 1],
        help="train : test : val.",
    )

    parser.add_argument(
        "--out-height",
        type=int,
        default=1080,
        help="Output image height. support 360, 480, 720",
    )

    parser.add_argument(
        "--seed", type=int, default=123, help="Seed for random number generating."
    )

    return parser.parse_args()


def resize_data(image_folder, annot_folder):
    image_list = os.listdir(image_folder)
    annot_list = os.listdir(annot_folder)

    name_list = []
    for image, annot in zip(image_list, annot_list):
        image_path = os.path.join(image_folder, image)
        with Image.open(image_path) as im:
            target_width = im.width / im.height * args.out_height
            if im.height != args.out_height:
                try:
                    im.resize((int(target_width), args.out_height)).save(image_path)
                except OSError:
                    print("[Data Corrupted] Can not resize {}.".format(image))
                    continue

        annot_path = os.path.join(annot_folder, annot)
        tree = ETree.parse(annot_path)
        root = tree.getroot()
        width = root.find("size").find("width")
        height = root.find("size").find("height")
        float_width = float(width.text)
        float_height = float(height.text)

        if root.find("object") is None:
            print("[Empty annot] No object found in {}.".format(annot))
        else:
            for obj in root.iter("object"):
                bndbox = obj.find("bndbox")

                xmin = float(bndbox.findtext("xmin")) / float_width * target_width
                ymin = float(bndbox.findtext("ymin")) / float_height * args.out_height
                xmax = float(bndbox.findtext("xmax")) / float_width * target_width
                ymax = float(bndbox.findtext("ymax")) / float_height * args.out_height

                bndbox.find("xmin").text = str(min(max(xmin, 0), float_width - 1))
                bndbox.find("ymin").text = str(min(max(ymin, 0), float_height - 1))
                bndbox.find("xmax").text = str(min(max(xmax, xmin), float_width))
                bndbox.find("ymax").text = str(min(max(ymax, ymin), float_height))

            name_list.append(".".join(annot.split(".")[:-1]))

        width.text = str(int(target_width))
        height.text = str(int(args.out_height))

        tree.write(annot_path)

    return name_list


def to_yolov5(folder_path):
    image_src_path = os.path.join(folder_path, ORIGIN_IMAGE_FOLDER)
    annot_src_path = os.path.join(folder_path, ORIGIN_LABEL_FOLDER)

    image_dst_path = os.path.join(folder_path, YOLO5_IMAGE_FOLDER)
    annot_dst_path = os.path.join(folder_path, YOLO5_LABEL_FOLDER)

    if not os.path.exists(image_dst_path):
        os.rename(image_src_path, image_dst_path)

    if not os.path.exists(annot_dst_path):
        os.mkdir(annot_dst_path)

    name_list = resize_data(image_dst_path, annot_src_path)

    with open(os.path.join(folder_path, "name_list.txt"), "w+") as f:
        for n in name_list:
            f.write(n + "\n")

    for annot in name_list:
        annot_path = os.path.join(annot_src_path, annot + ".xml")
        tree = ETree.parse(annot_path)
        root = tree.getroot()
        width = float(root.find("size").findtext("width"))
        height = float(root.find("size").findtext("height"))

        if root.find("object"):
            with open(os.path.join(annot_dst_path, annot + ".txt"), "w+") as f:
                for obj in root.iter("object"):
                    bndbox = obj.find("bndbox")
                    xmin = float(bndbox.findtext("xmin"))
                    ymin = float(bndbox.findtext("ymin"))
                    xmax = float(bndbox.findtext("xmax"))
                    ymax = float(bndbox.findtext("ymax"))

                    x = (xmin + xmax) / 2.0 / width
                    y = (ymin + ymax) / 2.0 / height
                    w = (xmax - xmin) / width
                    h = (ymax - ymin) / height

                    obj_name = obj.findtext("name")

                    try:
                        class_index = CLASSES.index(obj_name)
                        f.write(
                            " ".join([str(class_index), str(x), str(y), str(w), str(h)])
                            + "\n"
                        )
                    except ValueError:
                        pass


if __name__ == "__main__":
    args = parse_args()
    print("Start conversion...")
    dataset_root = os.path.expanduser(args.dji_roco_dir)

    if os.path.isdir(dataset_root):
        if all([os.path.isdir(os.path.join(dataset_root, path)) for path in SRC_LIST]):
            for s, d in zip(SRC_LIST, DES_LIST):
                os.rename(os.path.join(dataset_root, s), os.path.join(dataset_root, d))

        elif all(
            [os.path.isdir(os.path.join(dataset_root, path)) for path in DES_LIST]
        ):
            print("Converted. Do it again.")

        else:
            raise ValueError(
                ("{} do not contains DJI ROCO dataset.".format(dataset_root))
            )
    else:
        raise ValueError(("{} is not a valid directory.".format(dataset_root)))

    process_list = []

    for region_folder in DES_LIST:
        region_folder_path = os.path.join(dataset_root, region_folder)
        process = Process(target=to_yolov5, args=(region_folder_path,))

        process_list.append(process)

        process.start()
        print("[{}]Processing...".format(region_folder))

    for process in process_list:
        process.join()
    print("Completed.")

    sum_list = []
    for region_folder in DES_LIST:
        region_folder_path = os.path.join(dataset_root, region_folder)

        with open(os.path.join(region_folder_path, "name_list.txt"), "r") as f:
            for line in f.readlines():
                line = line.replace("\n", ".jpg\n")
                sum_list.append(
                    os.path.join(".", region_folder, YOLO5_IMAGE_FOLDER, line)
                )

    random.shuffle(sum_list)

    train_list, test_list, val_list = split_list(sum_list, args.split_ratio)

    train_file_path = os.path.join(dataset_root, "train.txt")
    val_file_path = os.path.join(dataset_root, "val.txt")
    test_file_path = os.path.join(dataset_root, "test.txt")

    with open(train_file_path, "w+") as f:
        f.writelines(train_list)

    with open(val_file_path, "w+") as f:
        f.writelines(test_list)

    with open(test_file_path, "w+") as f:
        f.writelines(val_list)

    yaml_out = {
        "train": train_file_path,
        "val": val_file_path,
        "test": test_file_path,
        "nc": len(CLASSES),
        "names": CLASSES,
    }

    with open("dataset.yaml", "w+") as f:
        yaml.dump(yaml_out, f)
        print(yaml.dump(yaml_out))

    print("Converted.")
