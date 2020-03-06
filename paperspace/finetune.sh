apt-get update
apt-get install -y libsm6 libxext6 libxrender-dev
pip install opencv-python

cd /storage/lacmus/

pip install . --user
python setup.py build_ext --inplace

mkdir logs/test
keras_retinanet/bin/train.py --epoch 1 --steps 100 --config config.ini --tensorboard-freq 1 --tensorboard-dir logs/test pascal /storage/data/LADDV4_Full >> logs/test/output.log
