There is a set of photos, reported by users, where retinaNet is finding people, while yolo is not 
(none of those included into train or validation, so it's orthgonal to reported metrics)
For those photos separate test of different configuration was conducted with following results:

|file		| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 
|-----------|---|---|---|---|---|---|---|---|---|---|
|12F01.jpeg	| + | - | - | - | - | - | - | + | - | + |
|12F02.jpeg | + | - | + | + | + | + | + | + | + | + |
|20F01.jpg 	| + | - | - | + | - | + | - | + | - | - |
|20F02.JPG 	| + | - | - | - | - | + | + | + | + | + |
|20F03.JPG  | + | - | + | + | + | + | + | + | + | + |
|20F04.jpg 	| + | - | - | + | + | + | + | + | + | + |
|-----------|---|---|---|---|---|---|---|---|---|---|
|sum        | 6 | 0 | 2 | 4 | 3 | 5 | 4 | 6 | 4 | 5 |

0 - lacmus app retina
1 - lacmus app yolo
2 - production network (from now on ran with yolov5 detect)
3 - production + augment (from now on augment assumes key "augment" to yolov5 during inference)
4 - train on ladd+ipsar in prod config 
5 - train on ladd+ipsar in prod config + augment
6 - yolov5s train on LADD+IPSAR on crops 
7 - same as 5 + augment 
8 - same as 6 in yolo large
9 - same as 7 + augment 