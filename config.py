# -*- coding: utf-8 -*-
"""
@Author: andy
@Contact: andy_viky@163.com
@Github: https://github.com/AndyandViky
@Csdn: https://blog.csdn.net/AndyViky
@File: config.py
@Time: 2020-02-15 00:05
@Desc: config.py
"""

import os

REPO_DIR = os.path.dirname(os.path.abspath(__file__))

# Local directory for datasets
DATASETS_DIR = os.path.join(REPO_DIR, 'datas')

RESULT_DIR = os.path.join(REPO_DIR, 'result')

# difference datasets config
# T, mix_threshold, algorithm_category, max_iter, dim, max_hy1f1_iter, gamma, z, u, v
DATA_PARAMS = {
    'big_data': [
        (7, 0.01, 0, 50, 3, 3000, 1, 1, 1, 0.01),
        (7, 0.01, 1, 50, 3, 3000, 1, 1, 1, 0.01),
    ],
    'big_data3': [
        (10, 0.01, 0, 50, 3, 3000, 1, 0.05, 1, 0.01),
        (10, 0.01, 1, 50, 3, 3000, 1, 0.05, 1, 0.01),
    ],
    'nyu': [
        (10, 0.01, 0, 50, 3, 5000, 1, 1, 0.1, 0.01),
        (10, 0.01, 1, 50, 3, 5000, 0.5, 1, 0.1, 0.01),
    ],
}

