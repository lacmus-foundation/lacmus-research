## Objective
This file contains comparison of mAP after one epoch of training keras_retinanet(resnet50) on different image sizes:

### Default
Standart keras_retinanet resizing, 1333x800  
Command:  
*keras_retinanet/bin/train.py --epoch 1 --config config.ini pascal ../../data/laddv4/full*

### 2000x1500
Resizing to about half of each side  
Command:  
*keras_retinanet/bin/train.py --epoch 1 --config config.ini --max-side 2000 -min-side 1500 pascal ../../data/laddv4/full*

### 3000x2250
About 3/4 of each side  
Command:  
*keras_retinanet/bin/train.py --epoch 1 --config config.ini --image-max-side 3000 --image-min-side 2250 pascal ../../data/laddv4/full*

### GridCrops
[Grid crops](https://github.com/lacmus-foundation/lacmus/blob/master/keras_retinanet/preprocessing/pascal_voc_grid_crops.py) of 2000x1500 size of full resolution  
Command:  
*keras_retinanet/bin/train.py --epoch 1 --config config.ini --no-resize pascal-grid-crops ../../data/laddv4/full --crop-width 2000 --crop-height 1500 --overlap-width 200 --overlap-height 200* 

### BBoxCrops
Crops taken around bounding boxes ([source code](https://github.com/prickly-u/lacmus/blob/balanced_crops/keras_retinanet/preprocessing/pascal_voc_balanced_crops.py)).
Crop sizes 1333x800 of full resolution  
Command:  
*keras_retinanet/bin/train.py --epoch 1 --snapshot-path snapshots/balanced_crops --weights snapshots/resnet50_base_best.h5 --config config.ini --batch-size 1  pascal-crops-balanced ../../data/laddv4/full --crop-width 1333 --crop-height 800 --negatives-per-positive 0* 

## Results

| Training Type |   mAP     |
| ------------- | --------- |
| Default       |  0.8040   |
| 2000x1500     |  0.9030   |
| 3000x2250     |  0.9341   |
| GridCrops     |  0.8637   |
| BboxCrops     |  0.9199   |

