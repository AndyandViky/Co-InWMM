# encoding: utf-8
"""
@author: andy
@contact: andy_viky@163.com
@github: https://github.com/AndyandViky
@csdn: https://blog.csdn.net/AndyViky
@file: plot.py
@time: 2019/6/9 下午8:20
@desc: plot
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_seg(data, labels, size, rgb_data=None, dep_data=None, nor_data=None):

    plt.xticks([])
    plt.yticks([])
    # plt.imshow(rgb_data / 255)
    # plt.show()
    # plt.imshow(dep_data / 255, cmap='gray')
    # plt.show()
    plt.imshow(nor_data)
    plt.show()

    colors = np.array([
        [255, 48, 48],
        [30, 144, 255],
        [255, 165, 0],
        [205, 201, 201],
        [50, 205, 50],
        [0, 191, 255],
        [139, 117, 0],
        [139, 126, 102],
    ])
    colors = colors / 255.0
    category = np.unique(labels)

    for i in range(len(category)):
        data[labels == category[i]] = colors[i]
    data = data.reshape(size)

    plt.imshow(data)
    plt.show()
