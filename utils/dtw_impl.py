import numpy as np

# def finalldist(s1, s2):
#     n, m = len(s1), len(s2)
#     accumulate_table = np.zeros((n, m))
#     dist = np.zeros((n, m))
#     for i in range(n):
#         for j in range(m):
#             vec = np.array(s1[i]) - np.array(s2[j])
#             dist[i][j] = np.inner(vec, vec)
#             if i == 0 and j == 0:
#                 accumulate_table[i][j] = dist[i][j]
#             elif i == 0 and j > 0:
#                 accumulate_table[i][j] = dist[i][j] + accumulate_table[i][j - 1]
#             elif i > 0 and j == 0:
#                 accumulate_table[i][j] = dist[i][j] + accumulate_table[i - 1][j]
#             else:
#                 accumulate_table[i][j] = dist[i][j] + min(accumulate_table[i][j - 1], accumulate_table[i - 1][j],
#                                                           accumulate_table[i - 1][j - 1])
#     return accumulate_table[i][j]

def finalldist(s1, s2):
    n, m = len(s1), len(s2)
    accumulate_table = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            dist = ((s1[i][0] - s2[j][0]) ** 2 + (s1[i][1] - s2[j][1]) ** 2 + (s1[i][2] - s2[j][2]) ** 2) ** 0.5
            if i == 0 and j == 0:
                accumulate_table[i][j] = dist
            elif i == 0 and j > 0:
                accumulate_table[i][j] = dist + accumulate_table[i][j - 1]
            elif i > 0 and j == 0:
                accumulate_table[i][j] = dist + accumulate_table[i - 1][j]
            else:
                accumulate_table[i][j] = dist + min(accumulate_table[i][j - 1], accumulate_table[i - 1][j], accumulate_table[i - 1][j - 1])
    return accumulate_table[i][j]
