import numpy as np
import matplotlib.pyplot as plt
from tools import read_predicted_boxes, read_ground_truth_boxes



def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """
    # YOUR CODE HERE
    if prediction_box[0] >= gt_box[2] or gt_box[0] >= prediction_box[2]:
        # If no overlap in x-dir
        return 0
    if prediction_box[1] >= gt_box[3] or gt_box[1] >= prediction_box[3]:
        # If no overlap in y-dir
        return 0

    # Compute intersection
    boxes = np.array([prediction_box, gt_box])

    right_box = np.argmax((prediction_box[0], gt_box[0]))
    x_max_right = boxes[right_box][2]
    x_min_right = boxes[right_box][0]
    x_inter = min(x_max_right, boxes[int(not right_box)][2]) - x_min_right
    
    upper_box = np.argmax((prediction_box[1], gt_box[1]))
    y_max_upper = boxes[upper_box][3]
    y_min_upper = boxes[upper_box][1]
    y_inter = min(y_max_upper, boxes[int(not upper_box)][3]) - y_min_upper
    
    area_intersection = x_inter*y_inter

    # Compute union as the total area minus the intersection
    tot_area = (prediction_box[2] - prediction_box[0]) \
                *(prediction_box[3]-prediction_box[1]) \
                +(gt_box[2] - gt_box[0]) \
                *(gt_box[3]-gt_box[1])
    union = tot_area - area_intersection
    iou = area_intersection/union
    assert iou >= 0 and iou <= 1
    return iou


def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    if (num_tp + num_fp) == 0:
        return 1
    return float(num_tp)/(num_tp + num_fp)


def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    if (num_tp + num_fn) == 0:
        return 0
    return float(num_tp)/(num_tp + num_fn)


def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """
    # Find all possible matches with a IoU >= iou threshold
    possible_matches = {}
    num_matches = {}
    ious = {}
    for pred_ind, pred_box in enumerate(prediction_boxes):
        num_matches[pred_ind] = 0
        for gt_ind, gt_box in enumerate(gt_boxes):
            iou = calculate_iou(pred_box, gt_box)
            if iou >= 0.5:
                if pred_ind in possible_matches:
                    possible_matches[pred_ind].append(gt_ind)
                    ious[pred_ind].append(iou)
                    num_matches[pred_ind] += 1
                else:
                    possible_matches[pred_ind] = [gt_ind]
                    ious[pred_ind] = [iou]
                    num_matches[pred_ind] = 1
        if pred_ind in num_matches:
            if num_matches[pred_ind] > 1:
                # If several matches, sort in descending order
                sorted_ious = sorted(ious[pred_ind], reverse=True)
                sorted_matches = []
                for iou in sorted_ious:
                    sort_ind = ious[pred_ind].index(iou)
                    sorted_matches.append(possible_matches[pred_ind][sort_ind])
                possible_matches[pred_ind] = sorted_matches
                ious[pred_ind] = sorted_ious

    iou_vals = list(ious.values())
    iou_sorted = sorted(iou_vals, reverse=True)

    used_gts = []
    used_preds = []
    pred_matched = []
    gt_matched = []
    
    for iou in iou_sorted:
        for pred_ind, iou_vals in ious.items():
            if pred_ind in used_preds:
                continue
            if num_matches[pred_ind] > 1:
                for iou_ind, current_iou in enumerate(iou_vals):
                    if current_iou == iou[0]:
                        gt_ind = possible_matches[pred_ind][iou_ind]
                        if gt_ind in used_gts:
                            continue
                        else:
                            pred_matched.append(prediction_boxes[pred_ind])
                            gt_matched.append(gt_boxes[gt_ind])
                            used_gts.append(gt_ind)
                            used_preds.append(pred_ind)
            else:
                if iou_vals == iou:
                    gt_ind = possible_matches[pred_ind]
                    if gt_ind in used_gts:
                        continue
                    else:
                        pred_matched.append(prediction_boxes[pred_ind])
                        gt_matched.append(gt_boxes[gt_ind])
                        used_gts.append(gt_ind)
                        used_preds.append(pred_ind)

        
  
    pred_matched = np.array(pred_matched)
    gt_matched = np.array(gt_matched, dtype=object)

    return pred_matched, gt_matched


def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, false_neg": int}
    """

    pred_matched, _ = get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold)
    
    true_pos = len(pred_matched)
    false_pos = len(prediction_boxes) - true_pos
    false_neg = len(gt_boxes) - true_pos

    return {"true_pos": true_pos, "false_pos": false_pos, "false_neg": false_neg}


def calculate_precision_recall_all_images(
    all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """

    true_pos = 0
    false_pos = 0
    false_neg = 0
    for (pred_boxes, gt_boxes) in zip(all_prediction_boxes, all_gt_boxes):
        image_result = calculate_individual_image_result(pred_boxes, gt_boxes, iou_threshold)
        true_pos += image_result["true_pos"]
        false_pos += image_result["false_pos"]
        false_neg += image_result["false_neg"]
        
    precision = calculate_precision(true_pos, false_pos, false_neg)
    recall = calculate_recall(true_pos, false_pos, false_neg)
    return (precision, recall)


def get_precision_recall_curve(
    all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold
):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        precisions, recalls: two np.ndarray with same shape.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    confidence_thresholds = np.linspace(0, 1, 500)
    precisions = []
    recalls = []
    for thresh in confidence_thresholds:
        pred_boxes = []
        for image_ind in range(len(confidence_scores)):
            pred_boxes.append([])
            for ind, prob in enumerate(confidence_scores[image_ind]):
                if prob > thresh:
                    pred_boxes[image_ind].append(all_prediction_boxes[image_ind][ind])
        precision, recall = calculate_precision_recall_all_images(
            pred_boxes, all_gt_boxes, iou_threshold)
        precisions.append(precision)
        recalls.append(recall)
    
    return np.array(precisions), np.array(recalls)


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    recall_levels = np.linspace(0, 1.0, 11)
    # YOUR CODE HERE
    smoothened_precision = []
    for r in recall_levels:
        right = precisions[recalls >= r]
        if not len(right):
            smoothened_precision.append(0)
            continue
        max_right = max(right)
        ind_p = np.where(precisions == max_right)[0][0]
        smoothened_precision.append(precisions[ind_p])
    smoothened_precision = np.array(smoothened_precision)
    average_precision = (1/11)*smoothened_precision.sum()
    return average_precision


def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)

    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions, recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)
