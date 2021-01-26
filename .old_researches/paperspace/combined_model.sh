#!/bin/bash

apt-get update
apt-get install -y libsm6 libxext6 libxrender-dev
pip install opencv-python

cd /storage/lacmus/

pip install . --user
python setup.py build_ext --inplace

mkdir logs/combined_model

mkdir logs/combined_model
keras_retinanet/bin/train.py --epoch 1 --steps 2000 --tensorboard-freq 10 --config config.ini --no-snapshots --no-random-transform --batch-size 1 --lr 0.0001 --image-min-side 1500 --image-max-side 2000 --regression-weight 0.5 --classification-weight 2.0 --optimizer-clipnorm 0.01 --tensorboard-dir logs/combined_model pascal /storage/data/LADDV4_Full >> logs/combined_model/output.log

