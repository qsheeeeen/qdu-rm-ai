#!/bin/bash

# Processing dataset. Generate file for training.
python3 roco2x.py

cd ../third_party/yolov5 || exit

core_count=$(nproc --all)
echo "CPU: ${core_count} cores."

# Traing model. batch-size shown for 10 GB devices.
python3 train.py \
--batch-size 26 \
--data ../../utils/dataset.yaml \
--img-size 608 608 \
--cfg ../../utils/model.yaml \
--hyp ../../utils/hyp.yaml \
--weights yolov5m.pt \
--worker ${core_count} \
--cache-images

# Test model.
python3 detect.py \
--source ../../image/ \
--output ../../image/result \
--weights runs/exp0/weights/best.pt \
--conf 0.4

# Export model to ONNX for inference.
export PYTHONPATH="$PWD"
python3 models/export.py --weights runs/exp0/weights/best.pt --img 608 --batch 1

# Copy ONNX file to target location.
cp runs/exp0/weights/best.onnx ../../mid/

cd ../../utils || exit
