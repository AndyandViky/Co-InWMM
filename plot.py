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
import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.pyplot import MultipleLocator

from config import RESULT_DIR, LOG_DIR


def plot_seg(data, labels, size, rgb_data=None, dep_data=None, nor_data=None, file_name='', save=False):

    # plt.imshow(rgb_data / 255)
    # plt.show()
    # plt.imshow(dep_data / 255, cmap='gray')
    # plt.show()
    # plt.imshow(nor_data)
    # plt.show()

    plt.axis('off')
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

    # plt.imshow(data, aspect='equal')
    # plt.show()

    if save:
        plt.imsave('{}/{}.png'.format(RESULT_DIR, file_name), data, dpi=500)


def plot_number_cluster(save=False):

    logger = open(os.path.join(LOG_DIR, "test.txt"))
    count = 0
    search = '['
    categorys = list()
    while True:
        line = logger.readline()
        if count == 0:
            count += 1
            continue
        count += 1
        if not line:
            break
        index = line.find(search, 0)
        category = line[index+1:-2].replace(' ', '')
        categorys.append(len(category))
    logger.close()

    categorys = np.array(categorys)

    ca_num = np.array([categorys[categorys == i].shape[0] for i in range(0, 11)])

    plt.tick_params(axis='both', which='major', labelsize=14)
    x_major_locator = MultipleLocator(1)
    y_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.plot(ca_num, '-|', ms=10, alpha=1, mfc='blue', label='CDP-WMM')
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.xlim(-0.5, 10.5)

    plt.xlabel('Number of components', fontsize=14)
    plt.ylabel('Number of images', fontsize=14)

    plt.legend()
    plt.show()

    if save:
        plt.savefig('{}/category_fig.png'.format(RESULT_DIR), dpi=200)


plot_number_cluster()

