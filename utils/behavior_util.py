import utils.posture_utils as pos_util
import utils.posture_config as pos_con
import utils.dtw_impl as dtw

def getDTWResult(behavior, templates):
    """
    計算DTW結果
    :param behavior: 行為序列
    :param templates: 行為模版
    :return: 最小距離
    """
    dists = []
    for template in templates:
        template_sequence = pos_util.digital2vector(template)
        dists.append(dtw.finalldist(behavior, template_sequence))
    return min(dists)

def recognizeBehavior(detected_sequence):
    """
    辨識行為
    :param detected_sequence: 偵測到的序列
    :return: 是否異常，dtw距離
    """
    abnormal_dist = getDTWResult(detected_sequence, pos_con.abnormal_template)# 取得行為序列與異常行為模板的最短距離
    normal_dist = getDTWResult(detected_sequence, pos_con.normal_template)# 取得行為序列與正常行為模板的最短距離
#    if abnormal_dist <= normal_dist and abnormal_dist < 13:
#        isAbnormal = True
#        dist = abnormal_dist
#    elif abnormal_dist >= normal_dist and normal_dist < 13:
#        isAbnormal = False
#        dist = normal_dist
#    else:
#        isAbnormal = 'unknow'
#        dist = normal_dist
        
    if abnormal_dist <= normal_dist and abnormal_dist < 13:
        isAbnormal = True
        dist = abnormal_dist

    else:
        isAbnormal = False
        dist = normal_dist
    
    return isAbnormal, dist
