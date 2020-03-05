pip install /storage/lacmus/ --user
python /storage/lacmus/setup.py build_ext --inplace


/storage/lacmus/keras_retinanet/bin/train.py --epoch 1 --steps 10 --config config.ini pascal /storage/lacmus/data/LADDV4_Full
