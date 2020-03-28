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
# T, mix_threshold, algorithm_category, max_iter, dim, max_hy1f1_iter
DATA_PARAMS = {
    'big_data': (7, 0.01, 0, 30, 3, 3000),
    'nyu': (30, 0.01, 1, 50, 3, 5000),
}
