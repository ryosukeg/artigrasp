### floating_arti

| object | split | success | success_once | angle_err(succ) | angle_err(all) | sim_dist(succ) |
|---|---|---|---|---|---|---|
| box | test | 0.508 | 0.600 | 0.179 | 0.549 | 0.009 |
| espressomachine | test | 0.793 | 0.859 | 0.216 | 0.348 | 0.005 |
| laptop | test | 0.480 | 0.547 | 0.176 | 0.573 | 0.013 |
| microwave | test | 0.500 | 0.550 | 0.245 | 0.619 | 0.005 |
| notebook | test | 0.036 | 0.036 | 0.038 | 0.982 | 0.001 |
| waffleiron | test | 0.619 | 0.695 | 0.276 | 0.539 | 0.025 |
| **mean** | test | 0.489 | 0.548 | 0.188 | 0.602 | 0.010 |

### floating_grasp

| object | split | success | pos_err(succ) | pos_err(all) | angle_err(succ) | angle_err(all) |
|---|---|---|---|---|---|---|
| box | test | 0.685 | 0.033 | 0.142 | 0.428 | 0.641 |
| espressomachine | test | 0.512 | 0.060 | 0.196 | 0.586 | 0.805 |
| ketchup | test | 0.478 | 0.025 | 0.272 | 0.380 | 1.232 |
| laptop | test | 0.981 | 0.024 | 0.030 | 0.334 | 0.344 |
| microwave | test | 0.248 | 0.087 | 0.284 | 0.743 | 1.207 |
| mixer | test | 0.693 | 0.030 | 0.151 | 0.428 | 0.845 |
| notebook | test | 0.879 | 0.026 | 0.070 | 0.393 | 0.726 |
| waffleiron | test | 0.724 | 0.036 | 0.119 | 0.408 | 0.825 |
| **mean** | test | 0.650 | 0.040 | 0.158 | 0.462 | 0.828 |

### fixed

| object | split | success | success_once | angle_err(succ) | angle_err(all) |
|---|---|---|---|---|---|
| box | test | 0.354 | 0.354 | 0.066 | 0.654 |
| espressomachine | test | 0.941 | 0.948 | 0.139 | 0.183 |
| ketchup | test | 0.333 | 0.378 | 0.247 | 0.788 |
| laptop | test | 0.493 | 0.493 | 0.087 | 0.560 |
| microwave | test | 0.800 | 0.850 | 0.222 | 0.354 |
| mixer | test | 0.730 | 0.748 | 0.260 | 0.466 |
| notebook | test | 0.418 | 0.455 | 0.088 | 0.578 |
| waffleiron | test | 0.543 | 0.610 | 0.134 | 0.511 |
| **mean** | test | 0.577 | 0.604 | 0.155 | 0.512 |

mean は物体ごとの値の単純平均(非有限値は除外)。'-' は値なし。
