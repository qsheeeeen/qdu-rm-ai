import argparse
import math
import os
import random


def parse_args():
    parser = argparse.ArgumentParser(
        description='Rearrange DJI ROCO dataset to VOC style.',
        epilog='Example: python dji_roco_to_voc.py --dji-roco-dir ~/Dataset/DJI ROCO/',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dji-roco-dir', type=str, default='~/Dataset/DJI ROCO/', help='dataset directory on disk')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print('Start conversion...')

    path = os.path.expanduser(args.dji_roco_dir)
    if not os.path.isdir(path) or not os.path.isdir(os.path.join(path, 'robomaster_Central China Regional Competition')) \
            or not os.path.isdir(os.path.join(path, 'robomaster_Final Tournament')) \
            or not os.path.isdir(os.path.join(path, 'robomaster_North China Regional Competition')) \
            or not os.path.isdir(os.path.join(path, 'robomaster_South China Regional Competition')):
        raise ValueError(('{} is not a valid directory, make sure it is present.'.format(path)))

    print('DJI ROCO root path: {}'.format(args.dji_roco_dir))

    group_list = os.listdir(path)

    for group in group_list:
        if not os.path.isdir(os.path.join(path, group)):
            group_list.remove(group)

    for group in group_list:
        print('Processing {}.'.format(group))
        group_path = os.path.join(path, group)

        print('Rename image -> JPEGImages')
        image_src_path = os.path.join(group_path, 'image')
        image_dst_path = os.path.join(group_path, 'JPEGImages')

        if not os.path.exists(image_dst_path):
            os.rename(image_src_path, image_dst_path)

        print('Rename image_annotation -> Annotations')
        annotation_src_path = os.path.join(group_path, 'image_annotation')
        annotation_dst_path = os.path.join(group_path, 'Annotations')

        if not os.path.exists(annotation_dst_path):
            os.rename(annotation_src_path, annotation_dst_path)

        print('Load list.')
        image_list = os.listdir(image_dst_path)
        annotation_list = os.listdir(annotation_dst_path)

        image_list.sort()
        annotation_list.sort()

        print('Check list.')
        for i in range(len(image_list)):
            if image_list[i][0:-4] == annotation_list[i][0:-4]:
                continue
            else:
                raise (RuntimeError, 'Unmatched label: {} & {}'.format(image_list[i], annotation_list[i]))
        print('Pass.')

        print('Get final list.')
        name_list = [image[0:-4] for image in image_list]

        random.shuffle(name_list)

        split_percent = 0.7
        split_index = math.floor(len(name_list) * split_percent)
        train_list = name_list[:split_index]
        test_list = name_list[split_index:]

        print('Create dir for Imagesets.')
        imagesets_path = os.path.join(group_path, 'ImageSets')
        if not os.path.exists(imagesets_path):
            os.mkdir(imagesets_path)

        imagesets_main_path = os.path.join(imagesets_path, 'Main')
        if not os.path.exists(imagesets_main_path):
            os.mkdir(imagesets_main_path)

        print('Write to file')
        with open(os.path.join(imagesets_main_path, 'trainval.txt'), 'w+') as f:
            for name in train_list:
                f.write(name + '\n')

        with open(os.path.join(imagesets_main_path, 'test.txt'), 'w+') as f:
            for name in test_list:
                f.write(name + '\n')

        print('Completed {}.'.format(group))
        print()

    print('Converted.')
