import argparse
import math
import os
import random
import xml.etree.ElementTree as ETree
from multiprocessing import Process

from PIL import Image

ORIGIN_IMAGE_FOLDER = 'image'
VOC_IMAGE_FOLDER = 'JPEGImages'
YOLO5_IMAGE_FOLDER = 'images'

ORIGIN_LABEL_FOLDER = 'image_annotation'
VOC_LABEL_FOLDER = 'Annotations'
YOLO5_LABEL_FOLDER = 'labels'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Rearrange DJI ROCO dataset to VOC style.',
        epilog='Example: python3 roco2x.py --dji-roco-dir ~/Dataset/DJI ROCO/')

    parser.add_argument('--dji-roco-dir', type=str, default='~/Dataset/DJI ROCO/',
                        help='dataset directory on disk')

    parser.add_argument('--target', type=str, default='yolov5',
                        help='Output format. support voc, yolov5, coco')

    parser.add_argument('--split-ratio', type=float, default=0.8,
                        help='Training set size :  dataset size.')

    parser.add_argument('--out-size', type=int, default=360,
                        help='Output image height. support 360, 480, 720')

    parser.add_argument('--seed', type=int, default=123,
                        help='Seed for random number generating.')

    return parser.parse_args()


def resize_data(image_folder, annotation_folder):
    image_list = os.listdir(image_folder)
    annotation_list = os.listdir(annotation_folder)

    name_list = []
    for image, annotation in zip(image_list, annotation_list):
        image_path = os.path.join(image_folder, image)
        with Image.open(image_path) as im:
            target_width = im.width / im.height * args.out_size
            if im.height != args.out_size:
                try:
                    im.resize((int(target_width), args.out_size)).save(image_path)
                except OSError:
                    print('[Data Corrupted] Can not resize {}. '.format(image))
                    continue

        annotation_path = os.path.join(annotation_folder, annotation)
        tree = ETree.parse(annotation_path)
        root = tree.getroot()
        width = root.find('size').find('width')
        height = root.find('size').find('height')
        float_width = float(width.text)
        float_height = float(height.text)

        if root.find('object') is None:
            print('[Empty annotation] No object found in {}.'.format(annotation))
        else:
            for obj in root.iter('object'):
                bndbox = obj.find('bndbox')

                xmin = float(bndbox.findtext('xmin')) / float_width * target_width
                ymin = float(bndbox.findtext('ymin')) / float_height * args.out_size
                xmax = float(bndbox.findtext('xmax')) / float_width * target_width
                ymax = float(bndbox.findtext('ymax')) / float_height * args.out_size

                bndbox.find('xmin').text = str(min(max(xmin, 0), float_width - 1))
                bndbox.find('ymin').text = str(min(max(ymin, 0), float_height - 1))
                bndbox.find('xmax').text = str(min(max(xmax, xmin), float_width))
                bndbox.find('ymax').text = str(min(max(ymax, ymin), float_height))

            name_list.append('.'.join(annotation.split('.')[:-1]))

        width.text = str(int(target_width))
        height.text = str(int(args.out_size))

        tree.write(annotation_path)

    return name_list


def to_voc(folder_path, folder_name):
    print('[{}]Processing...'.format(folder_name))

    image_src_path = os.path.join(folder_path, ORIGIN_IMAGE_FOLDER)
    annotation_src_path = os.path.join(folder_path, ORIGIN_LABEL_FOLDER)

    if not os.path.exists(image_src_path):
        image_src_path = os.path.join(folder_path, YOLO5_IMAGE_FOLDER)

    image_dst_path = os.path.join(folder_path, VOC_IMAGE_FOLDER)
    annotation_dst_path = os.path.join(folder_path, VOC_LABEL_FOLDER)

    if not os.path.exists(image_dst_path) and not os.path.exists(annotation_dst_path):
        os.rename(image_src_path, image_dst_path)
        os.rename(annotation_src_path, annotation_dst_path)

    name_list = resize_data(image_dst_path, annotation_dst_path)

    random.shuffle(name_list)

    split_index = math.floor(len(name_list) * args.split_ratio)
    train_list = name_list[:split_index]
    test_list = name_list[split_index:]
    print('[{}]Training set size: {}.'.format(folder_name, len(train_list)))
    print('[{}]Test set size: {}.'.format(folder_name, len(test_list)))

    print('[{}]Create dir for pairing.'.format(folder_name))
    imagesets_path = os.path.join(folder_path, 'ImageSets')
    if not os.path.exists(imagesets_path):
        os.mkdir(imagesets_path)

    imagesets_main_path = os.path.join(imagesets_path, 'Main')
    if not os.path.exists(imagesets_main_path):
        os.mkdir(imagesets_main_path)

    print('[{}]Write pairing to file.'.format(folder_name))
    with open(os.path.join(imagesets_main_path, 'trainval.txt'), 'w+') as f:
        for name in train_list:
            f.write(name + '\n')

    with open(os.path.join(imagesets_main_path, 'test.txt'), 'w+') as f:
        for name in test_list:
            f.write(name + '\n')

    print('[{}]Completed.'.format(folder_name))
    print()


CLASSES = ['armor', 'base', 'watcher', 'car']


def to_yolov5(folder_path, folder_name):
    print('[{}]Processing...'.format(folder_name))

    image_src_path = os.path.join(folder_path, ORIGIN_IMAGE_FOLDER)
    annotation_src_path = os.path.join(folder_path, ORIGIN_LABEL_FOLDER)

    if not os.path.exists(image_src_path):
        image_src_path = os.path.join(folder_path, VOC_IMAGE_FOLDER)

    if not os.path.exists(annotation_src_path):
        annotation_src_path = os.path.join(folder_path, VOC_LABEL_FOLDER)

    image_dst_path = os.path.join(folder_path, YOLO5_IMAGE_FOLDER)
    annotation_dst_path = os.path.join(folder_path, YOLO5_LABEL_FOLDER)

    if not os.path.exists(image_dst_path):
        os.rename(image_src_path, image_dst_path)

    if not os.path.exists(annotation_dst_path):
        os.mkdir(annotation_dst_path)

    name_list = resize_data(image_dst_path, annotation_src_path)

    for annotation in name_list:
        annotation_path = os.path.join(annotation_src_path, annotation + '.xml')
        tree = ETree.parse(annotation_path)
        root = tree.getroot()
        width = float(root.find('size').findtext('width'))
        height = float(root.find('size').findtext('height'))

        if root.find('object'):
            with open(os.path.join(annotation_dst_path, annotation + '.txt'), 'w+') as f:
                for obj in root.iter('object'):
                    bndbox = obj.find('bndbox')
                    xmin = float(bndbox.findtext('xmin'))
                    ymin = float(bndbox.findtext('ymin'))
                    xmax = float(bndbox.findtext('xmax'))
                    ymax = float(bndbox.findtext('ymax'))

                    x = xmin / width
                    y = ymin / height
                    w = (xmax - xmin) / width
                    h = (ymax - ymin) / height

                    obj_name = obj.findtext('name')

                    try:
                        class_index = CLASSES.index(obj_name)
                        f.write(' '.join([str(class_index), str(x), str(y), str(w), str(h)]) + '\n')
                    except ValueError:
                        pass
    #                 TODO: ADD symble link.

    print('[{}]Completed.'.format(folder_name))


def to_yolov5_prepare(dataset_dir):
    dirs = [
        os.path.join(dataset_dir, 'train'),
        os.path.join(dataset_dir, 'val'),
        os.path.join(dataset_dir, 'test'),
    ]

    for d in dirs:
        if not os.path.exists(d):
            os.mkdir(d)

    with open('roco.yaml', 'w+') as f:
        f.write('train: {}\n'.format(dirs[0]))
        f.write('val: {}\n'.format(dirs[1]))
        f.write('nc: {}\n'.format(len(CLASSES)))
        f.write('name: {}\n'.format(CLASSES))


if __name__ == '__main__':
    args = parse_args()
    print('Start conversion...')
    dataset_path = os.path.expanduser(args.dji_roco_dir)

    s_list = [
        os.path.join(dataset_path, 'robomaster_Central China Regional Competition'),
        os.path.join(dataset_path, 'robomaster_Final Tournament'),
        os.path.join(dataset_path, 'robomaster_North China Regional Competition'),
        os.path.join(dataset_path, 'robomaster_South China Regional Competition'),
    ]

    d_list = [
        os.path.join(dataset_path, 'central'),
        os.path.join(dataset_path, 'final'),
        os.path.join(dataset_path, 'north'),
        os.path.join(dataset_path, 'south'),
    ]

    if os.path.isdir(dataset_path):
        if all([os.path.isdir(group_path) for group_path in s_list]):
            for s, d in zip(s_list, d_list):
                os.rename(s, d)

        elif all([os.path.isdir(path) for path in d_list]):
            print('Converted. Do it again.')

        else:
            raise ValueError(('{} do not contains DJI ROCO dataset.'.format(dataset_path)))
    else:
        raise ValueError(('{} is not a valid directory, make sure it is present.'.format(dataset_path)))

    if args.target == 'yolov5':
        target = to_yolov5
        to_yolov5_prepare(dataset_path)
    elif args.target == 'voc':
        target = to_voc
    else:
        raise ValueError('Unknow target: {}'.format(args.target))

    data_path_list = os.listdir(dataset_path)

    process_list = []
    for sub_folder in data_path_list:
        sub_folder_path = os.path.join(dataset_path, sub_folder)
        if not os.path.isdir(sub_folder_path) or sub_folder in ['train', 'val', 'test']:
            continue

        process = Process(target=target, args=(sub_folder_path, sub_folder))

        process_list.append(process)
        process.start()

    for process in process_list:
        process.join()

    print('Converted.')
