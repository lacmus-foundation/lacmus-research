#!/bin/bash

apt-get update
apt-get install -y libsm6 libxext6 libxrender-dev
pip install opencv-python

cd /storage/lacmus/

pip install . --user

mkdir logs/retrain_800_1333

keras_retinanet/bin/train.py --weights ./snapshots/funetuning_e4.h5 --epoch 1 --tensorboard-freq 100 --config config.ini --batch-size 1 --optimizer-clipnorm 0.01 --tensorboard-dir logs/logs/retrain_800_1333 pascal /storage/data/LADDV4_Full >> logs/logs/retrain_800_1333/output.log

mkdir /artifacts/logs/retrain_800_1333
cp logs/logs/retrain_800_1333/* /artifacts/logs/retrain_800_1333/
cp snapshots/resnet50_pascal_01.h5 /artifacts/logs/retrain_800_1333/logs/finetuning_800_1333.h5  

