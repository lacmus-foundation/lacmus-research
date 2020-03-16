#!/bin/bash

apt-get update
apt-get install -y libsm6 libxext6 libxrender-dev
pip install opencv-python

cd /storage/lacmus/

pip install . --user

mkdir logs/retrain_800_1333_e5-6

keras_retinanet/bin/train.py --weights ./snapshots/finetuning_800_1333_e4.h5 --epoch 2 --tensorboard-freq 100 --config config.ini --batch-size 1 --optimizer-clipnorm 0.01 --tensorboard-dir logs/retrain_800_1333_e5-6 pascal /storage/data/LADDV4_Full >> logs/retrain_800_1333_e5-6/output.log

cp logs/retrain_800_1333_e3-4/* /artifacts
cp snapshots/resnet50_pascal_01.h5 /artifacts/finetuning_800_1333_e5.h5  
cp snapshots/resnet50_pascal_02.h5 /artifacts/finetuning_800_1333_e6.h5  

