### floating_arti

| object | split | success | success_once | angle_err(succ) | angle_err(all) | sim_dist(succ) |
|---|---|---|---|---|---|---|
| box | test | - | - | - | - | - |
| **mean** | test | - | - | - | - | - |

### floating_grasp

| object | split | success | pos_err(succ) | pos_err(all) | angle_err(succ) | angle_err(all) |
|---|---|---|---|---|---|---|
| box | test | - | - | - | - | - |
| **mean** | test | - | - | - | - | - |

### fixed

| object | split | success | success_once | angle_err(succ) | angle_err(all) |
|---|---|---|---|---|---|
| box | test | 0.354 | 0.354 | 0.066 | 0.654 |
| **mean** | test | 0.354 | 0.354 | 0.066 | 0.654 |

### failed / no metrics

| run | exit code | last log line |
|---|---|---|
| floating_arti_box_test | 1 | FileNotFoundError: ./data_all/pos_bias.txt not found. |
| floating_grasp_box_test | 1 | FileNotFoundError: ./data_all/pos_bias.txt not found. |

mean は物体ごとの値の単純平均(非有限値は除外)。'-' は値なし。
