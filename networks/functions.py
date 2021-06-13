import numpy as np
import pyximport
pyximport.install()
from compute_overlap import compute_overlap


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def evaluate_res(
    inference_res,
    iou_threshold=0.5,
    score_threshold=0.05,
    max_detections=100
):
    """ Evaluate a given dataset using a given model.
    # Arguments
        inference_res   : inference results for whole imageset List((target,prediction)),
            where targets {'boxes':tensor[4,n], 'labels':tenson[n]},
            prediction {'boxes':tensor[4,n], 'labels':tenson[n], scores: tensor[n]}
            example:

            [(({'boxes': tensor([[1321.8750,  274.6667, 1348.8750,  312.6667]]),
                'labels': tensor([1])},),
              [{'boxes': tensor([[1323.5446,  275.2711, 1350.2203,  315.9069],
                        [ 119.2671, 1227.5459,  171.1528, 1277.9830],
                        [ 240.5078, 1147.3656,  270.7879, 1205.0126],
                        [ 140.9097, 1231.9814,  173.9967, 1285.4724]]),
                'scores': tensor([0.9568, 0.3488, 0.1418, 0.0771]),
                'labels': tensor([1, 1, 1, 1])}]),
             (({'boxes': tensor([[ 798.7500, 1357.3334,  837.7500, 1396.6666],
                        [ 829.1250,  777.3333,  873.3750,  818.0000],
                        [ 886.5000,   34.6667,  916.5000,   77.3333]]),
                'labels': tensor([1, 1, 1])},),
              [{'boxes': tensor([[ 796.5808, 1354.9255,  836.5349, 1395.8972],
                        [ 828.8597,  777.9426,  872.5923,  819.8660],
                        [ 887.7839,   37.1435,  914.8092,   76.3933]]),
                'scores': tensor([0.9452, 0.8701, 0.8424]),
                'labels': tensor([1, 1, 1])}])]

        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
    """

    false_positives = np.zeros((0,))
    true_positives  = np.zeros((0,))
    scores          = np.zeros((0,))
    num_annotations = 0.0

    for i in range(len(inference_res)):
        detections           = inference_res[i][1][0]
        annotations          = inference_res[i][0][0]
        num_annotations     += inference_res[i][0][0]['labels'].shape[0]
        detected_annotations = []

        for d in range(detections['labels'].shape[0]):
            if detections['scores'][d].numpy() > score_threshold:
                scores = np.append(scores, detections['scores'][d].numpy())

                if inference_res[i][0][0]['labels'].shape[0] == 0: # no objects was there
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    continue

                overlaps            = compute_overlap (np.expand_dims(detections['boxes'][d].numpy().astype(np.double),axis=0),annotations['boxes'].numpy().astype(np.double))
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation][0]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives  = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)

    # F1@IoU
    plain_recall = np.sum(true_positives)/num_annotations
    plain_precision = np.sum(true_positives) / np.maximum(np.sum(true_positives) + np.sum(false_positives), np.finfo(np.float64).eps)
    F1 = 2*plain_precision*plain_recall/(plain_precision+plain_recall)


#     # sort by score
    indices         = np.argsort(-scores)
    false_positives = false_positives[indices]
    true_positives  = true_positives[indices]

#     # compute false positives and true positives
    false_positives = np.cumsum(false_positives)
    true_positives  = np.cumsum(true_positives)
#     # compute recall and precision
    recall    = true_positives / num_annotations
    precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

    # compute average precision
    average_precision  = _compute_ap(recall, precision)


    return (average_precision, F1)