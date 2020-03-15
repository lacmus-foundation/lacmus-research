#!/bin/bash

apt-get update
apt-get install -y libsm6 libxext6 libxrender-dev
pip install opencv-python

cd /storage/lacmus/

pip install . --user

mkdir logs/retrain_800_1333_e3-4

keras_retinanet/bin/train.py --weights ./snapshots/resnet50_pascal_01.h5 --epoch 1 --tensorboard-freq 100 --config config.ini --batch-size 1 --optimizer-clipnorm 0.01 --tensorboard-dir logs/retrain_800_1333_e3-4 pascal /storage/data/LADDV4_Full >> logs/retrain_800_1333_e3-4/output.log

cp logs/retrain_800_1333_e3-4/* /artifacts
cp snapshots/resnet50_pascal_01.h5 /artifacts/finetuning_800_1333_e3-4.h5  

