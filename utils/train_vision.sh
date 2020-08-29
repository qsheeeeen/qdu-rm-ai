#!/bin/bash

python3 roco2x.py

cd ../third_party/yolov5 || exit

# DJI
python3 train.py --batch 32 --data ../../utils/dataset.yaml --img-size 608 608 --cfg ../../utils/model.yaml --cache-images

export PYTHONPATH="$PWD"
python3 models/export.py --weights runs/exp0/weights/best.pt --img 608 --batch 1

cd ../../utils || exit
