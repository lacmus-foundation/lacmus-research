apt-get update
apt-get install -y libsm6 libxext6 libxrender-dev
pip install opencv-python

cd /storage/lacmus/

pip install . --user
python setup.py build_ext --inplace

mkdir logs/finetuning

mkdir logs/finetuning/base
keras_retinanet/bin/train.py --epoch 1 --steps 1000 --tensorboard-freq 10 --config config.ini --no-snapshots --no-random-transform --batch-size 2 --image-min-side 1500 --image-max-side 2000  --tensorboard-dir logs/finetuning/base pascal /storage/data/LADDV4_Full >> logs/finetuning/base/output.log

mkdir logs/finetuning/default_size
keras_retinanet/bin/train.py --epoch 1 --steps 1000 --tensorboard-freq 10 --config config.ini --no-snapshots --no-random-transform --batch-size 2  --tensorboard-dir logs/finetuning/default_size pascal /storage/data/LADDV4_Full >> logs/finetuning/default_size/output.log

mkdir logs/finetuning/batch_size_1
keras_retinanet/bin/train.py --epoch 1 --steps 1000 --tensorboard-freq 10 --config config.ini --no-snapshots --no-random-transform --batch-size 1 --image-min-side 1500 --image-max-side 2000 --tensorboard-dir logs/finetuning/batch_size_1 pascal /storage/data/LADDV4_Full >> logs/finetuning/batch_size_1/output.log

mkdir logs/finetuning/with_random_transform
keras_retinanet/bin/train.py --epoch 1 --steps 1000 --tensorboard-freq 10 --config config.ini --no-snapshots --batch-size 2 --image-min-side 1500 --image-max-side 2000 --tensorboard-dir logs/finetuning/with_random_transform pascal /storage/data/LADDV4_Full >> logs/finetuning/with_random_transform/output.log

mkdir logs/finetuning/lr_1e-4
keras_retinanet/bin/train.py --epoch 1 --steps 1000 --tensorboard-freq 10 --config config.ini --no-snapshots --no-random-transform --batch-size 2 --image-min-side 1500 --image-max-side 2000 --lr 0.0001 --tensorboard-dir logs/finetuning/lr_0001 pascal /storage/data/LADDV4_Full >> logs/finetuning/lr_0001/output.log

mkdir logs/finetuning/lr_1e-6
keras_retinanet/bin/train.py --epoch 1 --steps 1000 --tensorboard-freq 10 --config config.ini --no-snapshots --no-random-transform --batch-size 2 --image-min-side 1500 --image-max-side 2000 --lr 0.000001 --tensorboard-dir logs/finetuning/lr_1e-6 pascal /storage/data/LADDV4_Full >> logs/finetuning/lr_1e-6/output.log

mkdir logs/finetuning/gamma
keras_retinanet/bin/train.py --epoch 1 --steps 1000 --tensorboard-freq 10 --config config.ini --no-snapshots --no-random-transform --batch-size 2 --image-min-side 1500 --image-max-side 2000 --focal-gamma 5.0 --tensorboard-dir logs/finetuning/gamma pascal /storage/data/LADDV4_Full >> logs/finetuning/gamma/output.log

mkdir logs/finetuning/alpha
keras_retinanet/bin/train.py --epoch 1 --steps 1000 --tensorboard-freq 10 --config config.ini --no-snapshots --no-random-transform --batch-size 2 --image-min-side 1500 --image-max-side 2000 --focal-alpha 0.5 --tensorboard-dir logs/finetuning/alpha pascal /storage/data/LADDV4_Full >> logs/finetuning/alpha/output.log

mkdir logs/finetuning/weighted_loss
keras_retinanet/bin/train.py --epoch 1 --steps 1000 --tensorboard-freq 10 --config config.ini --no-snapshots --no-random-transform --batch-size 2 --image-min-side 1500 --image-max-side 2000 --regression-weight 0.5 --classification-weight 2.0  --tensorboard-dir logs/finetuning/weighted_loss pascal /storage/data/LADDV4_Full >> logs/finetuning/weighted_loss/output.log

mkdir logs/finetuning/clipnorm
keras_retinanet/bin/train.py --epoch 1 --steps 1000 --tensorboard-freq 10 --config config.ini --no-snapshots --no-random-transform --batch-size 2 --image-min-side 1500 --image-max-side 2000 --optimizer-clipnorm 0.01 --tensorboard-dir logs/finetuning/clipnorm pascal /storage/data/LADDV4_Full >> logs/finetuning/clipnorm/output.log

tar -czvf finetuning.tar.gz logs/finetuning/

