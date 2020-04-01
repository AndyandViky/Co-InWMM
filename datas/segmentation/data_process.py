# -*- coding: utf-8 -*-
"""
@Author: andy
@Contact: andy_viky@163.com
@Github: https://github.com/AndyandViky
@Csdn: https://blog.csdn.net/AndyViky
@File: data_process.py
@Time: 2020-04-01 15:27
@Desc: data_process.py
"""

import os
import scipy.io as scio
import pcl
import numpy as np
import matplotlib.pyplot as plt


def get_normal(cloud, num):

    mls = cloud.make_NormalEstimation()
    tree = cloud.make_kdtree()
    mls.set_SearchMethod(tree)
    mls.set_RadiusSearch(0.05)
    mls_points = mls.compute()

    normls = np.zeros((num, 4))
    count = 0
    for i in range(num):
        t = mls_points[i]
        if np.any(np.isnan(t)):
            count += 1
            # a little data point are nan, instead of common value.
            normls[i] = [0, 1, 0, 0]
        else:
            normls[i] = t

    print(count)
    normls = normls[:, :3]
    normls = normls.reshape((480, 640, 3))

    return normls


def file_name(file_dir):

    all_files = []
    for root, dirs, files in os.walk(file_dir):
        all_files.append(files)

    return all_files


def process():

    files = file_name('./toolbox/depths')[0]
    for index, name in enumerate(files):
        if index < 9:
            continue
        data = scio.loadmat('./toolbox/depths/{}'.format(name))['depImg'].reshape((-1, 3))
        cloud = pcl.PointCloud(data)

        normal = get_normal(cloud, data.shape[0])

        scio.savemat('./toolbox/normals/{}'.format(name), {'imgNormals': normal})
        plt.imshow(normal)
        plt.show()


if __name__ == '__main__':

    process()
