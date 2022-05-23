import math
import numpy as np
import utils.posture_config as pos_con


def getEachLabelBndData(name_list, obj_list):
    """
    對物件偵測器在current frame偵測到的所有物件進行對應label的整理
    :param name_list: label清單，儲存所有class name
    :param obj_list: 物件清單，儲存物件偵測器在current frame偵測到的所有物件
    :return: 每一label的bounding boxes的清單
    """
    total_data = {}
    for label in name_list:
        total_data[label] = []
    for obj in obj_list:
        for label in name_list:
            if obj[0] == label:
                total_data[label].append(obj[1])
    return total_data


def part_match(parts_list, body):
    """
    確認哪一部件屬於這隻魚
    :param parts_list: 某部件清單
    :param body: 某隻魚的bounding box
    :return: 部件的bounding box清單
    """
    tilapia_xmin, tilapia_ymin, tilapia_xmax, tilapia_ymax = body
    final_xy = []
    for j in range(len(parts_list)):
        xmin, ymin, xmax, ymax = parts_list[j]
        if (tilapia_xmax > int((xmax + xmin) / 2) > tilapia_xmin) and (tilapia_ymax > int((ymax + ymin) / 2) > tilapia_ymin):
            final_xy.append(parts_list[j])
    return final_xy


def getCompleteFish(all_bnd):
    """
    對所有部件進行匹配，確認是屬於哪一隻魚的部件
    :param all_bnd: 所有物件的bounding boxes
    :return:儲存每一隻魚所有資訊
    """
    fish = []
    for body in all_bnd['tilapia']:
        fish.append({
            'tilapia': [body],
            'analfin': part_match(all_bnd['analfin'], body),
            'dorsalfin': part_match(all_bnd['dorsalfin'], body),
            'fin': part_match(all_bnd['fin'], body),
            'head': part_match(all_bnd['head'], body),
            'eye': part_match(all_bnd['eye'], body),
            'mouth': part_match(all_bnd['mouth'], body),
            'tail': part_match(all_bnd['tail'], body),
        })
    return fish

def cart2Polar(center, x):
    """
    將直角座標轉成極座標
    :param center: 中心點
    :param x: 對應點
    :return: 半徑，角度
    """
    r = math.sqrt(math.pow(x[0] - center[0], 2) + math.pow(x[1] - center[1], 2))
    theta = math.atan2(x[1] - center[1], x[0] - center[0]) / math.pi * 180  # 轉換爲角度
    return r, theta


def quantizeDegrees(theta):
    """
    對角度進行量化
    :param theta: 角度
    :return: 量化後角度
    """
    if 22.5 <= theta < 67.5:
        return 45
    elif 67.5 <= theta < 112.5:
        return 90
    elif 112.5 <= theta < 157.5:
        return 135
    elif 157.5 <= theta < 202.5:
        return 180
    elif 202.5 <= theta < 247.5:
        return 225
    elif 247.5 <= theta < 292.5:
        return 270
    elif 292.5 <= theta < 337.5:
        return 315
    else:
        return 0


def getHeadDegrees(part_degrees, fish):
    """
    找出頭部的角度
    :param part_degrees: 各部件角度資訊
    :param fish: 單一隻魚的相關資訊，型態為字典
    :return: 頭部的量化角度
    """
    if part_degrees['head'] is None:
        # 不存在頭部部件， 嘗試以眼睛取代
        if part_degrees['eye'] is None:
            # 不存在眼睛部件， 嘗試以嘴巴取代
            if part_degrees['mouth'] is None:
                # 不存在嘴巴部件， 嘗試以尾巴反推
                if part_degrees['tail'] is None:
                    # 不存在尾巴部件， 嘗試以analfin與fin估測
                    af_c = np.array([int((fish['analfin'][0][0] + fish['analfin'][0][2]) / 2),
                                     int((fish['analfin'][0][1] + fish['analfin'][0][3]) / 2)])
                    f_c = np.array([int((fish['fin'][0][0] + fish['fin'][0][2]) / 2),
                                    int((fish['fin'][0][1] + fish['fin'][0][3]) / 2)])
                    _, h_theta = cart2Polar(af_c, f_c)
                else:
                    # 存在尾巴部件，嘗試以尾巴反推
                    h_theta = part_degrees['tail'] + 180
            else:
                # 存在嘴巴部件，嘗試以嘴巴取代
                h_theta = part_degrees['mouth']
        else:
            # 存在眼睛部件，嘗試以眼睛取代
            h_theta = part_degrees['eye']
    else:
        h_theta = part_degrees['head']
    if h_theta >= 360: # 進行同位角轉換
        h_theta -= 360
    h_theta = -h_theta # 影像直角座標與正常直角座標的Y軸方向相差180度，故將角度加上負號調整至正常直角座標
    if h_theta <= 0: # 進行同位角轉換
        h_theta += 360
    return quantizeDegrees(h_theta)


def checkPhaseSequence(pos):
    """
    檢查姿態屬於哪個相序
    :param pos: 部分姿態序
    :return: 對應的相序
    """
    if pos in pos_con.positive_phase_sequence:
        return 1 # 正相序
    elif pos in pos_con.negative_phase_sequence:
        return -1 # 逆向序
    else:
        return 0 # 無法定義


def getPartReference(degrees_list, part_degrees):
    """
    取得部件之間的關聯性
    :param degrees_list: 角度清單
    :param part_degrees:各部件角度資訊
    :return:部件之間的關聯性
    """
    degrees_list.sort()
    min_val1, min_val2, min_val3 = degrees_list[0], degrees_list[1], degrees_list[2]
    first_sym = ''
    second_sym = ''
    third_sym = ''
    for part in part_degrees:
        if min_val1 == part_degrees[part]:
            first_sym = pos_con.symbol_table[part]
        if min_val2 == part_degrees[part]:
            second_sym = pos_con.symbol_table[part]
        if min_val3 == part_degrees[part]:
            third_sym = pos_con.symbol_table[part]
    if (first_sym == 't' and second_sym == 'h') or (first_sym == 'h' and second_sym == 't') \
            or (first_sym == 'af' and second_sym == 'df') or (first_sym == 'f' and second_sym == 'df') \
            or (first_sym == 'df' and second_sym == 'af') or (first_sym == 'df' and second_sym == 'f'):
        if abs(min_val2 - min_val3) <= 180:
            part_ref = second_sym + '>' + third_sym
        else:
            part_ref = third_sym + '>' + second_sym
    else:
        if abs(min_val1 - min_val2) <= 180:
            part_ref = first_sym + '>' + second_sym
        else:
            part_ref = second_sym + '>' + first_sym
    return part_ref


def getPartDegrees(fish, main_box_name):
    """
    取得各部件的角度
    :param fish: 單一隻魚的相關資訊，型態為字典
    :param main_box_name: 主框的class name
    :return: 角度清單，各部件角度資訊
    """
    degrees_list = []
    part_degrees = {}
    body_c = np.array([int((fish[main_box_name][0][0] + fish[main_box_name][0][2]) / 2),
                       int((fish[main_box_name][0][1] + fish[main_box_name][0][3]) / 2)])
    for part in fish:
        if part != main_box_name and part != 'posture_vector':
            if len(fish[part]) != 0:
                part_c = np.array(
                    [int((fish[part][0][0] + fish[part][0][2]) / 2), int((fish[part][0][1] + fish[part][0][3]) / 2)])
                _, theta = cart2Polar(body_c, part_c)
                if theta < 0:
                    theta += 360
                if ((part == 'eye') and (part_degrees['head'] != None)):
                    degrees_list.append(4000)
                    part_degrees[part] = None
                elif ((part == 'mouth') and (part_degrees['eye'] != None)):
                    degrees_list.append(4000)
                    part_degrees[part] = None
                else:
                    degrees_list.append(theta)
                    part_degrees[part] = theta
            else:
                degrees_list.append(4000)
                part_degrees[part] = None
    return degrees_list, part_degrees


def decidePostureVector(fish, main_box_name):
    """
    決定姿態向量
    :param fish: 單一隻魚的相關資訊，型態為字典
    :param main_box_name: 主框的class name
    :return: 無回傳值，姿態向量已寫入fish字典內
    """
    degrees_list, part_degrees = getPartDegrees(fish, main_box_name)
    part_ref = getPartReference(degrees_list, part_degrees)
    gamma = checkPhaseSequence(part_ref)
    if gamma != 0:
        h_theta = getHeadDegrees(part_degrees, fish)
        x = math.cos(h_theta / 180 * math.pi)
        y = math.sin(h_theta / 180 * math.pi)
        fish['posture_vector'] = [round(x, 3), round(y, 3), gamma]
    else:
        fish['posture_vector'] = [0, 0, gamma]

def digital2vector(posture_sequence):
    vectors = []
    for posture in posture_sequence:
        vectors.append(pos_con.vector_table[str(posture)])
    return vectors
