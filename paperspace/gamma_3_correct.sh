#!/bin/bash

apt-get update
apt-get install -y libsm6 libxext6 libxrender-dev
pip install opencv-python

cd /storage/lacmus/

pip install . --user
python setup.py build_ext --inplace

mkdir logs/gamma_3_correct

mkdir logs/gamma_3_correct/gamma_3
keras_retinanet/bin/train.py --epoch 1 --steps 1000 --tensorboard-freq 10 --config config.ini --no-snapshots --no-random-transform --batch-size 2 --image-min-side 1500 --image-max-side 2000 --focal-gamma 3.0 --tensorboard-dir logs/gamma_3_correct/gamma_3 pascal /storage/data/LADDV4_Full >> logs/gamma_3_correct/gamma_3/output.log


tar -czvf /artifacts/gamma_3_correct.tar.gz logs/gamma_3_correct

