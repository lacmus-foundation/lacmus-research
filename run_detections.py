'''
This script allows to run inference on several snapshots simultaneously and place corresponding detections
in orderly folders structure.
Shoul be run from the directory containing 'lacmus' repository, with keras_retinanet newural network.
'''

import keras

from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu

import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

# use this to change which GPU to use
gpu = 0

# set the modified tf session as backend in keras
setup_gpu(gpu)

labels_to_names = {0: 'Pedestrian'}

models_paths = {
    "resnet50_liza_alert_v1": os.path.join('snapshots', 'resnet50_liza_alert_v1_interface.h5'),
    "finetuning_800_1333_e7": os.path.join('snapshots', 'finetuning_800_1333_e7_inference.h5')
}

# load retinanet model
models_dict = {key: models.load_model(models_paths[key], backbone_name='resnet50') for key in models_paths}

source_folder = "../TestImages/Source"
detections_folder = "../TestImages/Detections"

#for model_name in models_paths:
#    os.mkdir(os.path.join(detections_folder, model_name))

def run_detection_image(model, filepath, detections_folder):
    image = read_image_bgr(filepath)

    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    # print("processing time: ", time.time() - start)

    # correct for image scale
    boxes /= scale

    # visualize detections
    detections = 0
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.5:
            break

        detections += 1
        color = label_color(label)

        b = box.astype(int)
        draw_box(draw, b, color=color)

        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)
        #print("\nPedestrian!\n")

    if detections > 0:
        file, ext = os.path.splitext(filepath)
        image_name = file.split('/')[-1] + ext
        output_path = os.path.join(detections_folder, image_name)

        draw_conv = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
        cv2.imwrite(output_path, draw_conv)

image_sets = [f.name for f in os.scandir(source_folder) if f.is_dir()]
print(image_sets)

for image_set in image_sets:
    detections_folders =\
            {model_name: os.path.join(detections_folder, model_name, image_set) for model_name in models_paths}

    for detection_path in detections_folders.values():
        os.mkdir(detection_path)

    image_set_path = os.path.join(source_folder, image_set)
    images = [f for f in os.listdir(image_set_path)]

    for image in images:
        for model_name in models_paths:
            detection_path = detections_folders[model_name]
            model = models_dict[model_name]
            run_detection_image(model, os.path.join(image_set_path, image), detection_path)