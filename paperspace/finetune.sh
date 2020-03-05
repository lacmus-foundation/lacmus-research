#pip install --upgrade pip
pip3 install /storage/lacmus/ --user
python3 /storage/lacmus/setup.py build_ext --inplace


/storage/lacmus/keras_retinanet/bin/train.py --epoch 1 --steps 10 --config config.ini pascal /storage/lacmus/data/LADDV4_Full
