#!/bin/bash
cd ../third_party/yolov5 || exit

# DJI
python3 train.py --batch 32 --data ../../utils/dataset.yaml --img-size 608 608 --cfg ../../utils/model.yaml --cache-images

python3 models/export.py --weights runs/exp0/best.pt --img 608 --batch 1

cd ../../utils || exit
