import numpy as np

def caloverlap(input, ref):
    xmin0, ymin0, xmax0, ymax0 = input
    xmin1, ymin1, xmax1, ymax1 = ref
    xmin = min(xmin0, xmin1, xmax0, xmax1)
    ymin = min(ymin0, ymin1, ymax0, ymax1)
    xmax = max(xmin0, xmin1, xmax0, xmax1)
    ymax = max(ymin0, ymin1, ymax0, ymax1)
    temp = np.zeros((ymax - ymin, xmax - xmin))
    temp0 = temp.copy()
    temp0[ymin0 - ymin:ymax0 - ymin, xmin0 - xmin:xmax0 - xmin] = 1
    temp1 = temp.copy()
    temp1[ymin1 - ymin:ymax1 - ymin, xmin1 - xmin:xmax1 - xmin] = 1
    temp2 = temp0 + temp1
    temp3 = np.where(temp2 == 2, 1, 0)
    temp4 = np.sum(temp3) / temp3.size
    return temp4

def caloverlap01(input, ref):
    if input[0] < ref[0]:
        xmin0, ymin0, xmax0, ymax0 = ref
        xmin1, ymin1, xmax1, ymax1 = input
    else:
        xmin0, ymin0, xmax0, ymax0 = input
        xmin1, ymin1, xmax1, ymax1 = ref
    if xmin0 <= xmax1 and xmax1 <= xmax0:
        x = xmax0 - xmin1
    elif xmin1 <= xmin0 and xmax0 <= xmax1:
        x = xmax0 - xmin0
    elif xmin0 <= xmin1 and xmax1 <= xmax0:
        x = xmax1 - xmin1
    else:
        x = xmax1 - xmin0

    if ymin1 <= ymax0 and ymax0 <= ymax1:
        y = ymax0 - ymin1
    elif ymin1 <= ymin0 and ymax0 <= ymax1:
        y = ymax0 - ymin0
    elif ymin0 <= ymin1 and ymax1 <= ymax0:
        y = ymax1 - ymin1
    else:
        y = ymax1 - ymin0
    # temp = (x * y) / (((xmax0 - xmin0) * (ymax0 - ymin0)) + ((xmax1 - xmin1) * (ymax1 - ymin1)) - (x * y))
    return (x * y) / (((xmax0 - xmin0) * (ymax0 - ymin0)) + ((xmax1 - xmin1) * (ymax1 - ymin1)) - (x * y))

def calc_iou(gt_bbox, pred_bbox):
    '''
    This function takes the predicted bounding box and ground truth bounding box and
    return the IoU ratio
    '''
    x_topleft_gt, y_topleft_gt, x_bottomright_gt, y_bottomright_gt = gt_bbox
    x_topleft_p, y_topleft_p, x_bottomright_p, y_bottomright_p = pred_bbox
    if (x_topleft_gt > x_bottomright_gt) or (y_topleft_gt > y_bottomright_gt):
        raise AssertionError("Ground Truth Bounding Box is not correct")
    if (x_topleft_p > x_bottomright_p) or (y_topleft_p > y_bottomright_p):
        raise AssertionError("Predicted Bounding Box is not correct", x_topleft_p, x_bottomright_p, y_topleft_p,
                             y_bottomright_gt)
    # if the GT bbox and predcited BBox do not overlap then iou=0
    if (x_bottomright_gt < x_topleft_p):
        # If bottom right of x-coordinate  GT  bbox is less than or above the top left of x coordinate of  the predicted BBox
        return 0.0
    if (y_bottomright_gt < y_topleft_p):  # If bottom right of y-coordinate  GT  bbox is less than or above the top left of y coordinate of  the predicted BBox
        return 0.0
    if (x_topleft_gt > x_bottomright_p):  # If bottom right of x-coordinate  GT  bbox is greater than or below the bottom right  of x coordinate of  the predcited BBox
        return 0.0
    if (y_topleft_gt > y_bottomright_p):  # If bottom right of y-coordinate  GT  bbox is greater than or below the bottom right  of y coordinate of  the predcited BBox
        return 0.0

    GT_bbox_area = (x_bottomright_gt - x_topleft_gt + 1) * (y_bottomright_gt - y_topleft_gt + 1)
    Pred_bbox_area = (x_bottomright_p - x_topleft_p + 1) * (y_bottomright_p - y_topleft_p + 1)

    x_top_left = np.max([x_topleft_gt, x_topleft_p])
    y_top_left = np.max([y_topleft_gt, y_topleft_p])
    x_bottom_right = np.min([x_bottomright_gt, x_bottomright_p])
    y_bottom_right = np.min([y_bottomright_gt, y_bottomright_p])

    intersection_area = (x_bottom_right - x_top_left + 1) * (y_bottom_right - y_top_left + 1)

    union_area = (GT_bbox_area + Pred_bbox_area - intersection_area)

    return intersection_area / union_area

