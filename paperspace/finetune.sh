apt-get update
apt-get install -y libsm6 libxext6 libxrender-dev
pip install opencv-python

cd /storage/lacmus/

pip install . --user
python setup.py build_ext --inplace


keras_retinanet/bin/train.py --epoch 1 --steps 10 --config config.ini pascal /storage/lacmus/data/LADDV4_Full
