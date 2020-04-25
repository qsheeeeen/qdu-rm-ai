import os

from autogluon import ObjectDetection as task

data_root = '/home/qs/Dataset/DJI ROCO/robomaster_Central China Regional Competition'
# data_root = '/home/qs/Dataset/AutoGluon Tutorial/tiny_motorbike'
dataset_train = task.Dataset(data_root)
# dataset_train = task.Dataset(name='voc')

detector = task.fit(dataset_train)

dataset_test = task.Dataset(data_root, index_file_name='test')
test_map = detector.evaluate(dataset_test)
print("mAP on test dataset: {}".format(test_map[1][1]))

image = 'AllianceVsArtisans_BO2_2_0.jpg'
# image = '000467.jpg'
image_path = os.path.join(data_root, 'JPEGImages', image)

ind, prob, loc = detector.predict(image_path)
