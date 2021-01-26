#!/usr/bin/env python3

import os
import time
import argparse

import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore


def parse_args(args):
    parser = argparse.ArgumentParser(description='convert model')
    parser.add_argument(
        '--img_dir',
        help='path to the dir with images',
        type=str,
        required=True
    )
    parser.add_argument(
        '--img_list_path',
        help='path to the file with test images names',
        type=str,
        required=False
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
        '--device_str',
        help='Argument to pass as device name when loading network. Pass "all" to run all options',
        type=str,
        required=True,
        default="CPU"
    )
    parser.add_argument(
        '--print_detections',
        help='Whether to print detections',
        action='store_true'
    )

    return parser.parse_args(args)


def decode_openvino_detections(detections, input_shape = (800, 1333)):
    """
    Converts openvino detections to understandable format

    Parameters:
    detections: Detections obtained from net.infer() method.
    input_shape: This is required to scale the bounding boxes coordinates passed.

    Returns:
    boxes: The bounding box coordinates representing (xmin, ymin, xmax, ymax)
    scores: The confidence of the detections
    labels: The class of the object detected

    """
    detections = detections[:,:,detections[:,:,:,2].argsort()[0][0][::-1],:] # sort detections on score
    labels = detections[:,:,:,1].astype(int)
    scores = detections[:,:,:,2]
    boxes = detections[:,:,:,(3,4,5,6)] # in decimal
    # rescale to pixel
    boxes[:,:,:,(0,2)] = boxes[:,:,:,(0,2)]*input_shape[1]
    boxes[:,:,:,(1,3)] = boxes[:,:,:,(1,3)]*input_shape[0]

    return boxes[0], scores[0], labels[0]

def compute_resize_scale(image_shape, min_side=800, max_side=1333):
    """ Compute an image scale such that the image size is constrained to min_side and max_side.

    Args
        min_side: The image's min side will be equal to min_side after resizing.
        max_side: If after resizing the image's max side is above max_side, resize until the max side is equal to max_side.

    Returns
        A resizing scale.
    """
    (rows, cols, _) = image_shape

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side

    return scale

def resize_image(img, min_side=800, max_side=1200):
    """ Resize an image such that the size is constrained to min_side and max_side.

    Args
        min_side: The image's min side will be equal to min_side after resizing.
        max_side: If after resizing the image's max side is above max_side, resize until the max side is equal to max_side.

    Returns
        A resized image.
    """
    # compute scale to resize the image
    scale = compute_resize_scale(img.shape, min_side=min_side, max_side=max_side)

    # resize the image with the computed scale
    img = cv2.resize(img, None, fx=scale, fy=scale)

    return img, scale

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

def create_blank(image, w, h, color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in BGR"""
    r_image = np.zeros((h, w, 3), np.uint8)
    r_image[:] = color
    r_image[:image.shape[0],:image.shape[1],:image.shape[2]] = image
    return r_image

def print_detections(image_path, detections, scale):
    labels_to_names = {0: 'Pedestrian'}

    basename = os.path.basename(image_path)
    print(basename + ":")
    boxes, scores, labels = decode_openvino_detections(detections)
    print('bboxes:', boxes.shape)
    print('scores:', scores.shape)
    print('labels:', labels.shape)

    boxes /= scale
    objects_count = 0

    print("*" * 20)
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.5:
            break
        b = np.array(box.astype(int)).astype(int)
        # x1 y1 x2 y2
        print(f'{labels_to_names[label]}:')
        print(f'\tscore: {score}')
        print(f'\tbox: {b[0]} {b[1]} {b[2]} {b[3]}')
        objects_count = objects_count + 1
    print(f'found objects: {objects_count}')

def prepare_image(image, width, height):
    image, scale = resize_image(image)
    image = create_blank(image, width, height)
    image = preprocess_image(image)
    image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    image = np.expand_dims(image, axis=0)
    return image, scale

def measure_simple_inference(
        images_list, model_xml, model_bin, device_str, print_detections, input_key='input_1'):
    core = IECore()
    #core.set_config({"CPU_BIND_THREAD": "YES"}, "CPU")
    net = core.read_network(model=model_xml, weights=model_bin)
    net.batch_size = 1
    output = next(iter(net.outputs))

    shape = net.input_info[input_key].input_data.shape
    _, _, height, width = shape
    config = {}
    executable_net = core.load_network(network=net, config=config, device_name=device_str)

    load_time = 0.0
    preprocess_time = 0.0
    infer_time = 0.0
    latency_time = 0.0
    detections = dict()
    scales = dict()
    print('Running network, please wait...')
    start = time.time()
    for image_path in images_list:
        load_start = time.time()
        image = cv2.imread(image_path)
        load_end = time.time()
        load_time += load_end - load_start

        image, scales[image_path] = prepare_image(image, width, height)
        preprocess_end = time.time()
        preprocess_time += preprocess_end - load_end

        res = executable_net.infer(inputs={input_key: image})
        detections[image_path] = res[output]
        infer_end = time.time()
        infer_time += infer_end - preprocess_end
        latency_time += infer_end - load_start

    throughput_time = time.time() - start

    if print_detections:
        for image_path in detections:
            print_detections(image_path, detections[image_path], scales[image_path])
            print()

    img_count = len(images_list)
    print('latency: {} sec per image'.format(latency_time/img_count))
    print('throughput: {} images per sec'.format(img_count / throughput_time))
    print('avg load time: {} s'.format(load_time / img_count))
    print('avg preprocess time: {} s'.format(preprocess_time / img_count))
    print('avg inference time: {} s'.format(infer_time / img_count))


def main(args=None):
    args = parse_args(args)
    model_xml = args.xml
    model_bin = args.bin
    images_dir = args.img_dir
    device_str = args.device_str
    print_detections = args.print_detections

    if not args.img_list_path:
        images_names = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
    else:
        with open(args.img_list) as f:
            images_names = [img.strip() + '.jpg' for img in f.readlines()]

    images_list = [os.path.join(images_dir, img) for img in images_names]

    if device_str.lower() == 'all':
        for device in ['CPU', 'GPU', 'HETERO:GPU,CPU', 'HETERO:CPU,GPU', 'MULTI:CPU,GPU', 'MULTI:GPU,CPU']:
            print("device_name:", device)
            measure_simple_inference(images_list, model_xml, model_bin, device, print_detections)
            print()
    else:
        measure_simple_inference(images_list, model_xml, model_bin, device_str, print_detections)


if __name__ == '__main__':
    main()