import cv2
import numpy as np
import utils.overlap as ol


def drawBox(img, bnd, obj_id, color=(255, 0, 0), thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX):
    """
    Drawing a object bounding box and writing its ID
    :param img: image
    :param bnd: bonding box
    :param obj_id: bounding ID
    :param color: bonding box color
    :param thickness: bonding box thickness
    :param font: text font
    :return: None
    """
    xmin, ymin, xmax, ymax = bnd
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)
    cv2.putText(img, "ID:" + str(obj_id), (xmin, ymin - 10), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)


def drawBoxes(img, obj_list, color=(255, 0, 0), thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX):
    """
    Drawing all objects bounding boxes
    :param img: image
    :param obj_list: list of objects
    :param color: bonding box color
    :param thickness: bonding box thickness
    :param font: text font
    :return: None
    """
    for i, obj in enumerate(obj_list):
        drawBox(img, obj[1], i, color, thickness, font)


def drawTrace(img, target, max_point=10):
    """
    Drawing target tracking trace
    :param img: image
    :param target: tracking target
    :param max_point: tracking trace maximum points
    :return: None
    """
    centres = target["centre"]
    points = len(centres) if len(centres) < max_point else max_point
    if points > 1:
        for i in range(points - 1):
            start = centres[-(i + 1)]
            end = centres[-(i + 2)]
            cv2.line(img, start, end, (0, 255, 255), 2)


def getMaxValIndex(values, threshold):
    """
    Getting maximum value and its index value
    :param values: all values
    :param threshold: screening condition threshold
    :return: index value, maximum value
    """
    value = max(values)
    if value >= threshold:
        index = values.index(value)
    else:
        index = None
    return index, value


def getMinValIndex(values, threshold):
    """
    Getting minimum value and its index value
    :param values: all values
    :param threshold: screening condition threshold
    :return: index value, minimum value
    """
    value = min(values)
    if value <= threshold:
        index = values.index(value)
    else:
        index = None
    return index, value


def getCentreDist(target_bnd, obj_bnd):
    """
    Calculating distance of two object with centre points
    :param target_bnd: tracking target bounding box coordinate
    :param obj_bnd: Bounding box coordinate of tracking target
    :return: distance of two object's centre points
    """
    target_centre = ((target_bnd[2] + target_bnd[0]) / 2, (target_bnd[3] + target_bnd[1]) / 2)
    obj_centre = ((obj_bnd[2] + obj_bnd[0]) / 2, (obj_bnd[3] + obj_bnd[1]) / 2)
    return ((target_centre[0] - obj_centre[0]) ** 2 + (target_centre[1] - obj_centre[1]) ** 2) ** 0.5


def addTarget(targets, obj_list, post_frame):
    """
    Add new tracking target to targets list
    :param targets: all targets list
    :param obj_list: new targets list
    :param post_frame: Current frame number recorded for tracking start time
    :return:None
    """
    for obj in obj_list:
        targets.append({
            "position": [obj[1]],
            "centre": [(int((obj[1][0] + obj[1][2]) / 2), int((obj[1][1] + obj[1][3]) / 2))],
            "classname": [obj[0]],
            "state": "consider",
            "matchNum": None,
            "isTarcking": True,
            "predictNum": 0,
            "speed": [0],
            "start_frame": post_frame
        })


def targetUpdate(targets, update_list):
    """
    Updating the information with tracking targets
    :param targets: all targets list
    :param update_list: information list needed to update
    :return: None
    """
    for ul in update_list:
        target = targets[ul["target_match_num"]]
        obj = ul["obj"]
        target["position"].append(obj[1])
        target["centre"].append((int((obj[1][0] + obj[1][2]) / 2), int((obj[1][1] + obj[1][3]) / 2)))
        target["classname"].append(obj[0])
        target["predictNum"] = 0


def targetPredict(targets, predict_index_list, predcitMax=5):
    """
    Predicting the lost tracking targets
    :param targets: all targets list
    :param predict_index_list: index list needed to predict
    :return: None
    """
    for predict_index in predict_index_list:
        target = targets[predict_index]
        predict_times = target["predictNum"]
        if predict_times < predcitMax:
            if len(target["position"]) > 1:
                last_position = target["position"][-1]
                elder_position = target["position"][-2]
                last_w,  last_h = last_position[2] - last_position[0], last_position[3] - last_position[1]
                elder_w, elder_h = elder_position[2] - elder_position[0], elder_position[3] - elder_position[1]
                xmin = last_position[0] + int((elder_position[0] - last_position[0]) * 0.5)
                ymin = last_position[1] + int((elder_position[1] - last_position[1]) * 0.5)
                # xmin = last_position[0] + int((last_position[0] - elder_position[0]) * 0.5) + int(round((last_w - elder_w) * ((5 - predict_times + 1) / 5)))
                # ymin = last_position[1] + int((last_position[1] - elder_position[1]) * 0.5) + int(round((last_h - elder_h) * ((5 - predict_times + 1) / 5)))
                xmax = xmin + last_w
                ymax = ymin + last_h
                classname = target["classname"][-1]
                centre = (int((xmin + xmax) / 2), int((ymin + ymax) / 2))
            else:
                xmin, ymin, xmax, ymax = target["position"][-1]
                classname = target["classname"][-1]
                centre = target["centre"][-1]
            target["position"].append(np.array([xmin, ymin, xmax, ymax]))
            target["classname"].append(classname)
            target["centre"].append(centre)
            target["predictNum"] += 1
        else:
            target["isTarcking"] = False


def targetMatching(targets, objs):
    """
    Matching the tracking targets and detected objects
    :param targets: all targets list
    :param objs: Objects detected in current frame
    :return: data set to update，data set to add，data set of index to predict
    """
    update_set = []
    update_index_set = []
    add_set = []
    for j, obj in enumerate(objs):
        obj_bnd = obj[1]
        overlaps = []
        nearly = []
        for k, target in enumerate(targets):
            if target["isTarcking"]:
                t_bnd = target["position"][-1]
                overlaps.append(ol.calc_iou(t_bnd, obj_bnd))
                nearly.append(getCentreDist(t_bnd, obj_bnd))
            else:
                overlaps.append(0)
                nearly.append(210)
        overlap_index, overlap_value = getMaxValIndex(overlaps, 0.3)
        nearly_index, nearly_value = getMinValIndex(nearly, 200)
        if (overlap_index == nearly_index) and (overlap_index is not None):
            target_match_num = overlap_index
        elif (overlap_index == nearly_index) and (overlap_index is None):
            target_match_num = None
        else:
            if overlap_index is None:
                target_match_num = None
                # target_match_num = nearly_index
            else:
                target_match_num = overlap_index
        if target_match_num is None:
            add_set.append(obj)
        else:
            update_index_set.append(target_match_num)
            update_set.append({
                "target_match_num": target_match_num,
                "obj_num": j,
                "obj": obj
            })
    predict_index_set = [j for j, target in enumerate(targets) if target["isTarcking"] and (j not in update_index_set)]
    return update_set, add_set, predict_index_set

