#!/bin/bash
cd ../yolov5 || exit

# DJI
# python3 train.py --data ../train/roco_dataset.yaml --cache-images --img-size 416 416 --cfg models/roco_model.yaml

# BCCD
# python3 train.py --data ../BCCD/data.yaml --cache-images --img-size 416 416 --cfg ../BCCD/model.yaml

# COCO
python3 train.py --data data/coco.yaml

cd ../train || exit
