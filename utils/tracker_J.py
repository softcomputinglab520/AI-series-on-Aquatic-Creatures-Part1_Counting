import numpy as np
from .overlap import *


def targetUpdate(objs, targets, t_amount, post_frame):
    N = np.array([obj[0] for obj in objs])
    N1 = np.array([obj[1] for obj in objs])
    if len(targets) < t_amount:  # 如果追蹤數量不足
        if len(targets) == 0:  # 當追蹤數歸零時
            N_empty = 1
            N_t_all_min = 0
        elif len(targets) > 0:
            N_empty = 0
            temp = [x['trajectory'][-1] for x in targets]  # 取trajectory_list中所有目標物的最後一組追蹤資料
            N_t = np.array(temp)
            N_t_all = (N_t[:, 0] + N_t[:, 2] / 2, N_t[:, 1] + N_t[:, 3] / 2)  # 把list轉乘Array,計算出所有追蹤框的中心點
        for i in range(len(N)):  # N自動補上追蹤候選的魚，不足追蹤最高數目也沒關係，必須排除追同一隻
            if N_empty == 0:
                N_t_all_min = np.min(((N_t_all[0] - N[i, 0] - N[i, 2] / 2) ** 2 + (
                            N_t_all[1] - N[i, 1] - N[i, 3] / 2) ** 2) ** 0.5)  # 矩陣運算，每個候選框跟目標追蹤框中心的距離
            if N_t_all_min > min(N[i, 2] / 2, N[i, 3] / 2) or N_empty == 1:  # 不要重複追蹤
                targets.append({
                    'trajectory': [N[i, :]],
                    'all_save': [np.array([int(post_frame), 1, 0, 0, 0])],#frame,追蹤框是真還估測,追蹤框位移量,速度,角度
                    'classname': [N1[i]],
                    'state': 'consider'
                })
            if len(targets) == t_amount:
                break
    return N, N1


def targetUpdate01(objs, targets, t_amount, post_frame, main_box_name):
    """
    確認是否需要新增追蹤目標
    :param objs: 偵測器偵測到的主框
    :param targets: 追蹤目標暫存器
    :param t_amount: 追蹤數量上限
    :param post_frame: 現在的frame數
    :param main_box_name: 主框classname
    :return:主框資訊及主框classname
    """
    N = np.array([np.array([obj[main_box_name][0][0], obj[main_box_name][0][1], obj[main_box_name][0][2] - obj[main_box_name][0][0], obj[main_box_name][0][3]- obj[main_box_name][0][1]]) for obj in objs])
    N1 = np.array([main_box_name for _ in objs])
    if len(targets) < t_amount:  # 如果追蹤數量不足
        if len(targets) == 0:  # 當追蹤數歸零時
            N_empty = 1
            N_t_all_min = 0
        elif len(targets) > 0:
            N_empty = 0
            temp = [x['trajectory'][-1] for x in targets]  # 取trajectory_list中所有目標物的最後一組追蹤資料
            N_t = np.array(temp)
            N_t_all = (N_t[:, 0] + N_t[:, 2] / 2, N_t[:, 1] + N_t[:, 3] / 2)  # 把list轉乘Array,計算出所有追蹤框的中心點
        for i in range(len(N)):  # N自動補上追蹤候選的魚，不足追蹤最高數目也沒關係，必須排除追同一隻
            if N_empty == 0:
                N_t_all_min = np.min(((N_t_all[0] - N[i, 0] - N[i, 2] / 2) ** 2 + (
                            N_t_all[1] - N[i, 1] - N[i, 3] / 2) ** 2) ** 0.5)  # 矩陣運算，每個候選框跟目標追蹤框中心的距離
            if N_t_all_min > min(N[i, 2] / 2, N[i, 3] / 2) or N_empty == 1:  # 不要重複追蹤
                targets.append({
                    'trajectory': [N[i, :]],
                    'all_save': [np.array([int(post_frame), 1, 0, 0, 0])],#frame,追蹤框是真還估測,追蹤框位移量,速度,角度
                    'classname': [N1[i]],
                    'state': 'consider',
                    'fish_data': [objs[i]],
                    'frame_data': [int(post_frame)]
                })
                if len(targets) == t_amount:
                    break
    return N, N1


def targetMatch(N, N1, trajectory, all_save, class_name):
    target_tracking = trajectory[-1]
    class_name_temp = class_name[-1]
    xmin, ymin, xw, yh = target_tracking  # 上一個frame的追蹤結果
    A_overlap_C = 0  # 旗標
    C_C = all_save[-1][1]
    A_overlap1_temp = []
    bnd = [xmin, ymin, xmin + xw, ymin + yh]
    for k in range(len(N)):  # 在同一張0的矩陣畫上所有候選框的橢圓面積，即使重疊也均是1
        bnd1 = [N[k, 0], N[k, 1], N[k, 0] + N[k, 2], N[k, 1] + N[k, 3]]
        # A_overlap1 = caloverlap01(bnd1, bnd)
        A_overlap1 = calc_iou(bnd, bnd1)
        A_overlap1_temp.append(A_overlap1)
    if len(A_overlap1_temp) == 0:
        A_overlap1_max = 0
    else:
        A_overlap1_max = max(A_overlap1_temp)
    if A_overlap1_max > 0.3:  # 篩選是否重複追蹤
        distance = tuple([N[:, 0] + N[:, 2] / 2, N[:, 1] + N[:, 3] / 2])  # 儲存每個候選框的中心，未重疊面積的就是(0,0)
        distance_all = ((distance[0] - xmin - xw / 2) ** 2 + (
                    distance[1] - ymin - yh / 2) ** 2) ** 0.5  # 矩陣運算，每個候選框跟目標追蹤框中心的距離
        distance_min = np.min(distance_all)  # 基於面積有重疊，
        match_index = np.argmin(distance_all)
        target_tracking = N[match_index]
        class_name_temp1 = N1[match_index]
        target_color = (0, 255, 255)
        C_C = 0
    elif A_overlap_C == 0:  # 代表這個frmae沒有找到追蹤，動量修正
        distance_min = 0
        class_name_temp1 = class_name_temp
        if len(trajectory) > 1:
            target_tracking = [trajectory[-1][0] + trajectory[-1][0] - trajectory[len(trajectory) - 2][0] +
                               int(round((trajectory[-1][2] - trajectory[len(trajectory) - 2][2]) / 2)),
                               trajectory[-1][1] + trajectory[-1][1] - trajectory[len(trajectory) - 2][1] +
                               int(round((trajectory[-1][3] - trajectory[len(trajectory) - 2][3]) / 2)),
                               trajectory[-1][2],
                               trajectory[-1][3]]
        target_color = (0, 0, 0)  # 黑色
        C_C += 1
    return target_tracking, target_color, distance_min, C_C, class_name_temp1


def targetMatch01(N, N1, trajectory, all_save, class_name, fish_list, fish_data):
    """
    追蹤目標與偵測物件進行匹配
    :param N: 主框資訊
    :param N1: 主框classname
    :param trajectory: 追蹤主框資訊
    :param all_save: 追蹤其他資訊
    :param class_name: 主框類別
    :param fish_list: 當前frame的魚清單
    :param fish_data: 魚的資訊
    :return: 追蹤相關資訊
    """
    target_tracking = trajectory[-1]
    class_name_temp = class_name[-1]
    fish_data_temp = fish_data[-1]
    xmin, ymin, xw, yh = target_tracking  # 上一個frame的追蹤結果
    A_overlap_C = 0  # 旗標
    C_C = all_save[-1][1]
    A_overlap1_temp = []
    bnd = [xmin, ymin, xmin + xw, ymin + yh]
    for k in range(len(N)):  # 在同一張0的矩陣畫上所有候選框的橢圓面積，即使重疊也均是1
        bnd1 = [N[k, 0], N[k, 1], N[k, 0] + N[k, 2], N[k, 1] + N[k, 3]]
        # A_overlap1 = caloverlap01(bnd1, bnd)
        A_overlap1 = calc_iou(bnd, bnd1)
        A_overlap1_temp.append(A_overlap1)
    if len(A_overlap1_temp) == 0:
        A_overlap1_max = 0
    else:
        A_overlap1_max = max(A_overlap1_temp)
    if A_overlap1_max > 0.3:  # 篩選是否重複追蹤
        distance = tuple([N[:, 0] + N[:, 2] / 2, N[:, 1] + N[:, 3] / 2])  # 儲存每個候選框的中心，未重疊面積的就是(0,0)
        distance_all = ((distance[0] - xmin - xw / 2) ** 2 + (
                    distance[1] - ymin - yh / 2) ** 2) ** 0.5  # 矩陣運算，每個候選框跟目標追蹤框中心的距離
        distance_min = np.min(distance_all)  # 基於面積有重疊，
        match_index = np.argmin(distance_all)
        target_tracking = N[match_index]
        class_name_temp1 = N1[match_index]
        fish_data_temp1 = fish_list[match_index]
        if np.all(np.array(fish_data_temp1['posture_vector']) == 0):
            fish_data_temp1['posture_vector'] = fish_data_temp['posture_vector'].copy()
        target_color = (0, 255, 255)
        C_C = 0
    elif A_overlap_C == 0:  # 代表這個frmae沒有找到追蹤，動量修正
        distance_min = 0
        class_name_temp1 = class_name_temp
        if len(trajectory) > 1:
            target_tracking = [trajectory[-1][0] + trajectory[-1][0] - trajectory[len(trajectory) - 2][0] +
                               int(round((trajectory[-1][2] - trajectory[len(trajectory) - 2][2]) / 2)),
                               trajectory[-1][1] + trajectory[-1][1] - trajectory[len(trajectory) - 2][1] +
                               int(round((trajectory[-1][3] - trajectory[len(trajectory) - 2][3]) / 2)),
                               trajectory[-1][2],
                               trajectory[-1][3]]
        target_color = (0, 0, 0)  # 黑色
        C_C += 1
        fish_data_temp1 = {
            'tilapia': [np.array(
                [int(target_tracking[0]), int(target_tracking[1]), int(target_tracking[0] + target_tracking[2]),
                 int(target_tracking[1] + target_tracking[3])])],
            'analfin': [],
            'dorsalfin': [],
            'fin': [],
            'head': [],
            'eye': [],
            'mouth': [],
            'tail': [],
            'posture_vector': fish_data_temp['posture_vector'].copy()
        }
    return target_tracking, target_color, distance_min, C_C, class_name_temp1, fish_data_temp1

