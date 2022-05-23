import utils.posture_config as pos_con
import utils.posture_utils as pos_util
from utils import dtw_impl as dtw_impl
import numpy as np
import time

digital_pos_sequence = pos_con.abnormal_template[0]
detected_sequence = pos_util.digital2vector(digital_pos_sequence)

template = pos_con.abnormal_template[1]
template_sequence = pos_util.digital2vector(template)

n, m = len(digital_pos_sequence), len(template_sequence)

accumulate_table = np.zeros((n, m))

dist = np.zeros((n, m))
for i in range(n):
    for j in range(m):
        vec = np.array(detected_sequence[i]) - np.array(template_sequence[j])
        dist[i][j] = np.inner(vec, vec)
        if i == 0 and j == 0:
            accumulate_table[i][j] = dist[i][j]
        elif i == 0 and j > 0:
            accumulate_table[i][j] = dist[i][j] + accumulate_table[i][j - 1]
        elif i > 0 and j == 0:
            accumulate_table[i][j] = dist[i][j] + accumulate_table[i-1][j]
        else:
            accumulate_table[i][j] = dist[i][j] + min(accumulate_table[i][j - 1], accumulate_table[i - 1][j], accumulate_table[i - 1][j - 1])

dist1 = accumulate_table[i][j]
start = time.time()
dist2 = dtw_impl.finalldist(detected_sequence, template_sequence)
print(time.time() - start)

# s1 = np.array(detected_sequence)
# s2 = np.array(template_sequence)
start = time.time()
accumulate_table01 = np.zeros((n, m))
for i in range(n):
    for j in range(m):
        dist01 = ((detected_sequence[i][0] - template_sequence[j][0]) ** 2 + (detected_sequence[i][1] - template_sequence[j][1]) ** 2 + (detected_sequence[i][2] - template_sequence[j][2]) ** 2) ** 0.5
        if i == 0 and j == 0:
            accumulate_table01[i][j] = dist01
        elif i == 0 and j > 0:
            accumulate_table01[i][j] = dist01 + accumulate_table01[i][j - 1]
        elif i > 0 and j == 0:
            accumulate_table01[i][j] = dist01 + accumulate_table01[i-1][j]
        else:
            accumulate_table01[i][j] = dist01 + min(accumulate_table01[i][j - 1], accumulate_table01[i - 1][j], accumulate_table01[i - 1][j - 1])
dist3 = accumulate_table[i][j]
print(time.time() - start)
