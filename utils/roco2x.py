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
CLASSES = ['armor', 'base', 'watcher', 'car']


def split_list(source_list, split_ratio):
    split_index1 = math.floor(len(source_list) * split_ratio[0] / sum(split_ratio))
    split_index2 = math.floor(len(source_list) * sum(split_ratio[:2]) / sum(split_ratio))

    train = source_list[:split_index1]
    test = source_list[split_index1:split_index2]
    val = source_list[split_index2:]

    return train, test, val


def parse_args():
    parser = argparse.ArgumentParser(
        description='Rearrange DJI ROCO dataset to VOC style.',
        epilog='Example: python3 roco2x.py --dji-roco-dir ~/Dataset/DJI ROCO/')

    parser.add_argument('--dji-roco-dir', type=str, default='~/Dataset/DJI ROCO/',
                        help='dataset directory on disk')

    parser.add_argument('--target', type=str, default='yolov5',
                        help='Output format. support voc, yolov5')

    parser.add_argument('--split-ratio', nargs='+', type=int, default=[7, 2, 1],
                        help='train : test : val.')

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

    train_list, test_list, val_list = split_list(name_list, args.split_ratio)

    print('[{}]Training set size: {}.'.format(folder_name, len(train_list)))
    print('[{}]Test set size: {}.'.format(folder_name, len(test_list)))
    print('[{}]Val set size: {}.'.format(folder_name, len(val_list)))

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

    with open(os.path.join(imagesets_main_path, 'val.txt'), 'w+') as f:
        for name in val_list:
            f.write(name + '\n')

    print('[{}]Completed.'.format(folder_name))
    print()




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

    with open(os.path.join(folder_path, 'name_list.txt'), 'w+') as f:
        for n in name_list:
            f.write(n + '\n')

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

                    x = (xmin + xmax) / 2.0 / width
                    y = (ymin + ymax) / 2.0 / height
                    w = (xmax - xmin) / width
                    h = (ymax - ymin) / height

                    obj_name = obj.findtext('name')

                    try:
                        class_index = CLASSES.index(obj_name)
                        f.write(' '.join([str(class_index), str(x), str(y), str(w), str(h)]) + '\n')
                    except ValueError:
                        pass

    print('[{}]Completed.'.format(folder_name))


if __name__ == '__main__':
    converted = False
    args = parse_args()
    print('Start conversion...')
    dataset_path = os.path.expanduser(args.dji_roco_dir)

    s_list = [
        'robomaster_Central China Regional Competition',
        'robomaster_Final Tournament',
        'robomaster_North China Regional Competition',
        'robomaster_South China Regional Competition',
    ]

    d_list = [
        'central',
        'final',
        'north',
        'south',
    ]

    if os.path.isdir(dataset_path):
        if all([os.path.isdir(group_path) for group_path in s_list]):
            for s, d in zip(s_list, d_list):
                os.rename(os.path.join(dataset_path, s), os.path.join(dataset_path, d))

        elif all([os.path.isdir(os.path.join(dataset_path, path)) for path in d_list]):
            converted = True
            print('Converted. Do it again.')

        else:
            raise ValueError(('{} do not contains DJI ROCO dataset.'.format(dataset_path)))
    else:
        raise ValueError(('{} is not a valid directory, make sure it is present.'.format(dataset_path)))

    if args.target == 'yolov5':
        target = to_yolov5
    elif args.target == 'voc':
        target = to_voc
    else:
        raise ValueError('Unknow target: {}'.format(args.target))

    data_path_list = os.listdir(dataset_path)

    process_list = []

    for sub_folder in d_list if converted else s_list:
        sub_folder_path = os.path.join(dataset_path, sub_folder)
        process = Process(target=target, args=(sub_folder_path, sub_folder))

        process_list.append(process)
        process.start()

    for process in process_list:
        process.join()

    if args.target == 'yolov5':
        sum_list = []
        for sub_folder in d_list:
            sub_folder_path = os.path.join(dataset_path, sub_folder)

            with open(os.path.join(sub_folder_path, 'name_list.txt')) as f:
                for line in f.readlines():
                    line = line.replace('\n', '.jpg\n')
                    sum_list.append(os.path.join('.', sub_folder, YOLO5_IMAGE_FOLDER, line))

        train_list, test_list, val_list = split_list(sum_list, args.split_ratio)

        with open(os.path.join(dataset_path, 'train.txt'), 'w+') as f:
            f.writelines(train_list)

        with open(os.path.join(dataset_path, 'val.txt'), 'w+') as f:
            f.writelines(test_list)

        with open(os.path.join(dataset_path, 'test.txt'), 'w+') as f:
            f.writelines(val_list)

    print('Converted.')
