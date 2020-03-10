#!/bin/bash

apt-get update
apt-get install -y libsm6 libxext6 libxrender-dev
pip install opencv-python

cd /storage/lacmus/

pip install . --user
python setup.py build_ext --inplace

mkdir logs/combined_model_e2

keras_retinanet/bin/train.py --snapshot ./snapshots/resnet50_pascal_01.h5 --initial-epoch 2 --epoch 1 --tensorboard-freq 100 --config config.ini --no-random-transform --batch-size 1 --lr 0.0001 --image-min-side 1500 --image-max-side 2000 --regression-weight 0.5 --classification-weight 2.0 --optimizer-clipnorm 0.01 --tensorboard-dir logs/combined_model_e2 pascal /storage/data/LADDV4_Full >> logs/combined_model_e2/output.log

mkdir /artifacts/combined_model_e2
cp logs/combined_model_e2/* /artifacts/combined_model_e2/
cp snapshots/resnet50_pascal_02.h5 /artifacts/combined_model_e2/funetuning_e2.h5  

