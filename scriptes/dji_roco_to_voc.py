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
                        help='Training set size :  dataset size.')

    parser.add_argument('--out-size', type=int, default=360,
                        help='Output image height. support 360, 480, 720')

    return parser.parse_args()


def to_voc(floder_path, floder_name):
    print('[{}]Processing...'.format(floder_name))
    print('[{}]Rename image -> JPEGImages.'.format(floder_name))
    image_src_path = os.path.join(floder_path, 'image')
    image_dst_path = os.path.join(floder_path, 'JPEGImages')

    if not os.path.exists(image_dst_path):
        os.rename(image_src_path, image_dst_path)

    print('[{}]Rename image_annotation -> Annotations.'.format(floder_name))
    annotation_src_path = os.path.join(floder_path, 'image_annotation')
    annotation_dst_path = os.path.join(floder_path, 'Annotations')

    if not os.path.exists(annotation_dst_path):
        os.rename(annotation_src_path, annotation_dst_path)

    print('[{}]Load list.'.format(floder_name))
    image_list = os.listdir(image_dst_path)
    annotation_list = os.listdir(annotation_dst_path)

    print('[{}]Check pairing...'.format(floder_name))
    image_list.sort()
    annotation_list.sort()

    if not len(image_list) == len(annotation_list):
        raise RuntimeError('[{}]Images and annotations should have the same size.'.format(floder_name))

    for i in range(len(image_list)):
        if image_list[i].split('.')[0] != annotation_list[i].split('.')[0]:
            raise (RuntimeError, 'Unmatched label: {} & {}'.format(image_list[i], annotation_list[i]))
    print('[{}]Pass.'.format(floder_name))

    name_list = []
    print('[{}]Resize Image...'.format(floder_name))
    print('[{}]Check annotation...'.format(floder_name))
    for image, annotation in zip(image_list, annotation_list):
        image_path = os.path.join(image_dst_path, image)
        with Image.open(image_path) as im:
            target_width = im.width / im.height * args.out_size
            if im.height != args.out_size:
                try:
                    im.resize((int(target_width), args.out_size)).save(image_path)
                except OSError:
                    print('[{}][Data Corrupted] Can not resize {}. '.format(floder_name, image))
                    continue

        annotation_path = os.path.join(annotation_dst_path, annotation)
        tree = ETree.parse(annotation_path)
        root = tree.getroot()
        width = root.find('size').find('width')
        height = root.find('size').find('height')
        float_width = float(width.text)
        float_height = float(height.text)

        if root.find('object') is None:
            print('[{}][Annotation Dropped] No object found in {}. '.format(floder_name, annotation))
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
    print('[{}]All Done.'.format(floder_name))

    random.shuffle(name_list)

    split_index = math.floor(len(name_list) * args.split_ratio)
    train_list = name_list[:split_index]
    test_list = name_list[split_index:]
    print('[{}]Training set size: {}.'.format(floder_name, len(train_list)))
    print('[{}]Test set size: {}.'.format(floder_name, len(test_list)))

    print('[{}]Create dir for pairing.'.format(floder_name))
    imagesets_path = os.path.join(floder_path, 'ImageSets')
    if not os.path.exists(imagesets_path):
        os.mkdir(imagesets_path)

    imagesets_main_path = os.path.join(imagesets_path, 'Main')
    if not os.path.exists(imagesets_main_path):
        os.mkdir(imagesets_main_path)

    print('[{}]Write pairing to file.'.format(floder_name))
    with open(os.path.join(imagesets_main_path, 'trainval.txt'), 'w+') as f:
        for name in train_list:
            f.write(name + '\n')

    with open(os.path.join(imagesets_main_path, 'test.txt'), 'w+') as f:
        for name in test_list:
            f.write(name + '\n')

    print('[{}]Completed.'.format(floder_name))
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
    for sub_folder in dir_list:
        sub_folder_path = os.path.join(path, sub_folder)
        if not os.path.isdir(sub_folder_path):
            continue

        process = Process(target=to_voc, args=(sub_folder_path, sub_folder))
        process_list.append(process)
        process.start()

    for process in process_list:
        process.join()

    print('Converted.')
