#!/usr/bin/env python

import os
import time
import argparse

import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore
import json


def parse_args(args):
    parser = argparse.ArgumentParser(description='convert model')
    parser.add_argument(
        '--img',
        help='path to image',
        type=str,
        required=True
    )
    parser.add_argument(
        '--bin',
        help='path to bin openVINO inference model',
        type=str,
        required=True
    )
    parser.add_argument(
        '--xml',
        help='path to xml model sheme',
        type=str,
        required=True
    )
    parser.add_argument(
        '--count',
        help='iference count',
        type=int,
        required=False,
        default=4
    )
    return parser.parse_args(args)

def preprocess_image(x, mode='caffe'):
    """ Preprocess an image by subtracting the ImageNet mean.

    Args
        x: np.array of shape (None, None, 3) or (3, None, None).
        mode: One of "caffe" or "tf".
            - caffe: will zero-center each color channel with
                respect to the ImageNet dataset, without scaling.
            - tf: will scale pixels between -1 and 1, sample-wise.

    Returns
        The input with the ImageNet mean subtracted.
    """
    # mostly identical to "https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py"
    # except for converting RGB -> BGR since we assume BGR already

    # covert always to float32 to keep compatibility with opencv
    x = x.astype(np.float32)

    if mode == 'tf':
        x /= 127.5
        x -= 1.
    elif mode == 'caffe':
        x[..., 0] -= 103.939
        x[..., 1] -= 116.779
        x[..., 2] -= 123.68

    return x

def main(args=None):
    args=parse_args(args)

    model_xml = args.xml
    model_bin = args.bin
    img_fn = args.img
    predict_count = args.count
    
    print("initialize OpenVino...")
    OpenVinoIE = IECore()
    print("available devices: ", OpenVinoIE.available_devices)
    
    OpenVinoIE.set_config({"CPU_BIND_THREAD": "YES"}, "CPU")

    print("loading model...")
    net = IENetwork(model=model_xml, weights=model_bin)
    config = {}
    OutputLayer = next(iter(net.outputs))
    OpenVinoExecutable = OpenVinoIE.load_network(network=net, config=config, device_name="CPU")

    input_blob = 'data_2'
    net.batch_size = 1
    _, _, h, w = net.inputs[input_blob].shape
    print(f'model input shape: {net.inputs[input_blob].shape}')


    # load images
    image = cv2.imread(img_fn)
    image = cv2.resize(image, (h, w))
    image = preprocess_image(image)

    image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    image = np.expand_dims(image, axis=0)

    print(f'make {predict_count} predictions:')

    for _ in range(0, predict_count):
        start_time = time.time()
        res = OpenVinoExecutable.infer(inputs={input_blob: image})
        print("\t{} s".format(time.time() - start_time))

if __name__ == '__main__':
    main()