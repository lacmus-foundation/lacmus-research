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
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
    """
    # gather all detections and annotations
#     all_detections, all_inferences = \
#         _get_detections(generator, model, score_threshold=score_threshold, max_detections=max_detections, save_path=save_path)
#     all_annotations    = _get_annotations(generator)
#     average_precisions = {}

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


    return average_precision