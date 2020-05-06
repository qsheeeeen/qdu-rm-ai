import argparse
import math
import os
import random
import xml.etree.ElementTree as ETree
from multiprocessing import Process

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description='Rearrange DJI ROCO dataset to VOC style.',
        epilog='Example: python3 dji_roco_to_voc.py --dji-roco-dir ~/Dataset/DJI ROCO/')

    parser.add_argument('--dji-roco-dir', type=str, default='~/Dataset/DJI ROCO/',
                        help='dataset directory on disk')

    parser.add_argument('--split-ratio', type=float, default=0.8,
                        help='Traing set size : test set size.')

    parser.add_argument('--out-size', type=int, default=720,
                        help='Output image height. support 360, 480, 720, 1080')

    return parser.parse_args()


def process_subfolder(dir_path, d):
    print('[{}]Processing...'.format(d))
    print('[{}]Rename image -> JPEGImages.'.format(d))
    image_src_path = os.path.join(dir_path, 'image')
    image_dst_path = os.path.join(dir_path, 'JPEGImages')

    if not os.path.exists(image_dst_path):
        os.rename(image_src_path, image_dst_path)

    print('[{}]Rename image_annotation -> Annotations.'.format(d))
    annotation_src_path = os.path.join(dir_path, 'image_annotation')
    annotation_dst_path = os.path.join(dir_path, 'Annotations')

    if not os.path.exists(annotation_dst_path):
        os.rename(annotation_src_path, annotation_dst_path)

    print('[{}]Load list.'.format(d))
    image_list = os.listdir(image_dst_path)
    annotation_list = os.listdir(annotation_dst_path)

    print('[{}]Check pairing...'.format(d))
    image_list.sort()
    annotation_list.sort()

    if not len(image_list) == len(annotation_list):
        raise RuntimeError('[{}]Images and annotations should have the same size.'.format(d))

    for i in range(len(image_list)):
        if image_list[i].split('.')[0] == annotation_list[i].split('.')[0]:
            continue
        else:
            raise (RuntimeError, 'Unmatched label: {} & {}'.format(image_list[i], annotation_list[i]))
    print('[{}]Pass.'.format(d))

    name_list = []
    print('[{}]Resize Image...'.format(d))
    print('[{}]Check annotation...'.format(d))
    for image, annotation in zip(image_list, annotation_list):
        image_path = os.path.join(image_dst_path, image)
        with Image.open(image_path) as im:
            target_width = im.width / im.height * args.out_size
            if im.height != args.out_size:
                try:
                    im.resize((int(target_width), args.out_size)).save(image_path)
                except OSError:
                    print('[{}][Data Corrupted] Can not resize {}. '.format(d, image))

        annotation_path = os.path.join(annotation_dst_path, annotation)
        tree = ETree.parse(annotation_path)
        root = tree.getroot()
        width = root.find('size').find('width')
        height = root.find('size').find('height')

        if root.find('object') is None:
            print('[{}][Annotation Dropped] No object found in {}. '.format(d, annotation))
        else:
            for obj in root.iter('object'):
                bndbox = obj.find('bndbox')
                bndbox.find('xmin').text = str(float(bndbox.findtext('xmin')) / float(width.text) * target_width)
                bndbox.find('ymin').text = str(float(bndbox.findtext('ymin')) / float(height.text) * args.out_size)
                bndbox.find('xmax').text = str(float(bndbox.findtext('xmax')) / float(width.text) * target_width)
                bndbox.find('ymax').text = str(float(bndbox.findtext('ymax')) / float(height.text) * args.out_size)
            name_list.append(annotation.split('.')[0])

        width.text = str(int(target_width))
        height.text = str(int(args.out_size))

        tree.write(annotation_path)
    print('[{}]Done.'.format(d))
    print('[{}]Done.'.format(d))

    random.shuffle(name_list)

    split_ratio = args.split_ratio
    split_index = math.floor(len(name_list) * split_ratio)
    train_list = name_list[:split_index]
    test_list = name_list[split_index:]
    print('[{}]Training set size: {}.'.format(d, len(train_list)))
    print('[{}]Test set size: {}.'.format(d, len(test_list)))

    print('[{}]Create dir for pairing.'.format(d))
    imagesets_path = os.path.join(dir_path, 'ImageSets')
    if not os.path.exists(imagesets_path):
        os.mkdir(imagesets_path)

    imagesets_main_path = os.path.join(imagesets_path, 'Main')
    if not os.path.exists(imagesets_main_path):
        os.mkdir(imagesets_main_path)

    print('[{}]Write pairing to file.'.format(d))
    with open(os.path.join(imagesets_main_path, 'trainval.txt'), 'w+') as f:
        for name in train_list:
            f.write(name + '\n')

    with open(os.path.join(imagesets_main_path, 'test.txt'), 'w+') as f:
        for name in test_list:
            f.write(name + '\n')

    print('[{}]Completed.'.format(d))
    print()


if __name__ == '__main__':
    args = parse_args()
    print('Start conversion...')

    path = os.path.expanduser(args.dji_roco_dir)

    s_central = os.path.join(path, 'robomaster_Central China Regional Competition')
    s_final = os.path.join(path, 'robomaster_Final Tournament')
    s_north = os.path.join(path, 'robomaster_North China Regional Competition')
    s_south = os.path.join(path, 'robomaster_South China Regional Competition')

    d_central = os.path.join(path, 'central')
    d_final = os.path.join(path, 'final')
    d_north = os.path.join(path, 'north')
    d_south = os.path.join(path, 'south')

    if os.path.isdir(path):
        if os.path.isdir(s_central) or os.path.isdir(s_final) or os.path.isdir(s_north) or os.path.isdir(s_south):
            os.rename(s_central, d_central)
            os.rename(s_final, d_final)
            os.rename(s_north, d_north)
            os.rename(s_south, d_south)

        elif os.path.isdir(d_central) and os.path.isdir(d_final) and os.path.isdir(d_north) and os.path.isdir(d_south):
            print('Converted. Do it again.')
        else:
            raise ValueError(('{} do not contains DJI ROCO dataset.'.format(path)))
    else:
        raise ValueError(('{} is not a valid directory, make sure it is present.'.format(path)))

    dir_list = os.listdir(path)

    process_list = []
    for d in dir_list:
        dir_path = os.path.join(path, d)
        if not os.path.isdir(dir_path):
            continue

        process_list.append(Process(target=process_subfolder, args=(dir_path, d)))

    for process in process_list:
        process.start()

    for process in process_list:
        process.join()

    print('Converted.')
