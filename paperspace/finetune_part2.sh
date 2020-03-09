#!/bin/bash

apt-get update
apt-get install -y libsm6 libxext6 libxrender-dev
pip install opencv-python

cd /storage/lacmus/

pip install . --user
python setup.py build_ext --inplace

mkdir logs/finetuning_part2

mkdir logs/finetuning_part2/batch_size_1
keras_retinanet/bin/train.py --epoch 1 --steps 2000 --tensorboard-freq 10 --config config.ini --no-snapshots --no-random-transform --batch-size 1 --image-min-side 1500 --image-max-side 2000 --tensorboard-dir logs/finetuning_part2/batch_size_1 pascal /storage/data/LADDV4_Full >> logs/finetuning_part2/batch_size_1/output.log

mkdir logs/finetuning_part2/lr_1e-4
keras_retinanet/bin/train.py --epoch 1 --steps 1000 --tensorboard-freq 10 --config config.ini --no-snapshots --no-random-transform --batch-size 2 --image-min-side 1500 --image-max-side 2000 --lr 0.0001 --tensorboard-dir logs/finetuning_part2/lr_1e-4 pascal /storage/data/LADDV4_Full >> logs/finetuning_part2/lr_1e-4/output.log

mkdir logs/finetuning_part2/gamma_3
keras_retinanet/bin/train.py --epoch 1 --steps 1000 --tensorboard-freq 10 --config config.ini --no-snapshots --no-random-transform --batch-size 2 --image-min-side 1500 --image-max-side 2000 --focal-gamma 5.0 --tensorboard-dir logs/finetuning_part2/gamma_3 pascal /storage/data/LADDV4_Full >> logs/finetuning_part2/gamma_3/output.log


tar -czvf finetuning2.tar.gz logs/finetuning_part2

