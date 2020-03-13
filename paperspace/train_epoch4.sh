#!/bin/bash

apt-get update
apt-get install -y libsm6 libxext6 libxrender-dev
pip install opencv-python

cd /storage/lacmus/

pip install . --user

mkdir logs/combined_model_e4

keras_retinanet/bin/train.py --weights ./snapshots/resnet50_pascal_01.h5 --epoch 1 --tensorboard-freq 100 --config config.ini --batch-size 1 --image-min-side 1500 --image-max-side 2000 --lr 0.000001 --optimizer-clipnorm 0.01 --tensorboard-dir logs/combined_model_e4 pascal /storage/data/LADDV4_Full >> logs/combined_model_e4/output.log

mkdir /artifacts/combined_model_e4
cp logs/combined_model_e4/* /artifacts/combined_model_e4/
cp snapshots/resnet50_pascal_01.h5 /artifacts/combined_model_e4/funetuning_e4.h5  

