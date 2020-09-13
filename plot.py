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
import io
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio

from math import *
from matplotlib.pyplot import MultipleLocator

from config import RESULT_DIR, LOG_DIR
from utils import file_name


def plot_seg(data, labels, size, rgb_data=None, dep_data=None, nor_data=None, root='', file_name='', save=False):

    if rgb_data is not None:
        plt.imshow(rgb_data / 255)
        plt.show()
    if dep_data is not None:
        plt.imshow(dep_data / 255, cmap='gray')
        plt.show()
    if nor_data is not None:
        plt.imshow(nor_data)
        plt.show()

    plt.axis('off')
    colors = np.array([
        [255, 48, 48],
        [30, 144, 255],
        [50, 205, 50],
        [205, 201, 201],
        [255, 165, 0],
        [0, 191, 255],
        [139, 117, 0],
        [139, 126, 102],
    ])
    colors = colors / 255.0
    category = np.unique(labels)

    for i in range(len(category)):
        data[labels == category[i]] = colors[i]
    data = data.reshape(size)

    if save:
        plt.imsave('{}/{}.png'.format(root, file_name), data, dpi=500)
    else:
        plt.imshow(data, aspect='equal')
        plt.show()


# import scipy.io as scio
# data = scio.loadmat('./datas/segmentation/toolbox/nyu.mat')['images']
# for i in range(data.shape[3]):
#     plt.axis('off')
#     plt.imsave('{}/rgb/nyu{}.png'.format(RESULT_DIR, i+1), data[:, :, :, i], dpi=500)


def plot_number_cluster(save=False):

    logger = open(os.path.join(LOG_DIR, "log.txt"))
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
    w_num = np.array([0, 0, 160, 460, 560, 200, 50, 10, 0, 0, 0])
    plt.tick_params(axis='both', which='major', labelsize=14)
    x_major_locator = MultipleLocator(1)
    y_major_locator = MultipleLocator(100)
    ax = plt.gca()
    ax.plot(ca_num, '-s', ms=6, alpha=1, mfc='blue', label='Co-InWMM')
    ax.plot(w_num, '-d', c='black', ms=6, alpha=1, mfc='black', label='WMM')
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.xlim(-0.5, 10.5)

    plt.xlabel('Number of components', fontsize=14)
    plt.ylabel('Number of images', fontsize=14)

    plt.legend(fontsize=13, markerscale=1.5)
    leg = ax.get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontweight='bold')

    if save:
        plt.savefig('{}/fig/category_fig.eps'.format(RESULT_DIR), dpi=200, format='eps')
    else:
        plt.show()


def plot_3d(dataset='syn_data1', save=False):

    fig = plt.figure(figsize=(7.4, 5.8))
    ax = fig.add_subplot(111, projection='3d')
    gca = fig.gca(projection='3d')

    # Make data
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = 1 * np.outer(np.cos(u), np.sin(v))
    y = 1 * np.outer(np.sin(u), np.sin(v))
    z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the surface
    ax.plot_surface(x, y, z, color='#ffffff', alpha=0, edgecolors='#999999', linewidth=0.1)

    data = scio.loadmat('./datas/{}.mat'.format(dataset))
    labels = data['z'].reshape(-1)

    if dataset == 'syn_data1':
        ax.view_init(azim=-50)
        # ax.set_title('Group 2', fontsize=20)
        data = data['data'].reshape((-1, 3))
        first = data[labels == 1]
        av_f = [1, 0, 0]
        second = data[labels == 2]
        av_s = [0, 1, 0]
        third = data[labels == 3]
        av_t = [0, 0, 1]

        ax.scatter3D(first[:, 2], first[:, 1], first[:, 0], c=first[:, 0], s=10, cmap='Blues', label='cluster 1', marker='*')
        gca.plot([0, av_f[0]], [0, av_f[1]], [0, av_f[2]], c='b', linewidth=0.5)
        gca.plot([0, -av_f[0]], [0, av_f[1]], [0, av_f[2]], c='b', linewidth=0.5)
        ax.scatter3D(second[:, 2], second[:, 1], second[:, 0], c=second[:, 0], s=10, cmap='Oranges', label='cluster 2', marker='o')
        gca.plot([0, av_s[0]], [0, av_s[1]], [0, av_s[2]], c='orange', linewidth=0.5)
        gca.plot([0, av_s[0]], [0, -av_s[1]], [0, av_s[2]], c='orange', linewidth=0.5)
        ax.scatter3D(third[:, 2], third[:, 1], third[:, 0], c=third[:, 1], s=10, cmap='Reds', label='cluster 3', marker='+')
        gca.plot([0, av_t[0]], [0, av_t[1]], [0, av_t[2]], c='r', linewidth=0.5)
        gca.plot([0, av_t[0]], [0, av_t[1]], [0, -av_t[2]], c='r', linewidth=0.5)
        ax.legend(bbox_to_anchor=(0.14, 1), markerscale=2.3, fontsize=13)
        # ax.legend(bbox_to_anchor=(0.81, 1.03), markerscale=1.3, fontsize=13)
    # big_data
    else:
        ax.view_init(azim=-50)
        # ax.set_title('Group 1', fontsize=20)
        data = scio.loadmat('./datas/{}.mat'.format(dataset))['data']
        first = data[labels == 1][:300]
        av_f = [1, 0, 0]
        second = data[labels == 2][:300]
        av_s = [0, 1, 0]
        third = data[labels == 3][:300]
        av_t = [0, 0, 1]
        four = data[labels == 4][:300]
        av_fo = [0.25, -0.25, 0.25]

        ax.scatter3D(first[:, 0], first[:, 1], first[:, 2], c=first[:, 1], s=10, cmap='Blues', label='cluster 1', marker='v')
        gca.plot([0, av_f[0]], [0, av_f[1]], [0, av_f[2]], c='b', linewidth=0.5)
        gca.plot([0, -av_f[0]], [0, av_f[1]], [0, av_f[2]], c='b', linewidth=0.5)
        ax.scatter3D(second[:, 0], second[:, 1], second[:, 2], c=second[:, 0], s=10, cmap='Reds', label='cluster 2', marker='x')
        gca.plot([0, av_s[0]], [0, av_s[1]], [0, av_s[2]], c='r', linewidth=0.5)
        gca.plot([0, av_s[0]], [0, -av_s[1]], [0, av_s[2]], c='r', linewidth=0.5)
        ax.scatter3D(third[:, 0], third[:, 1], third[:, 2], c=third[:, 2], s=10, cmap='Oranges', label='cluster 3', marker='*')
        gca.plot([0, av_t[0]], [0, av_t[1]], [0, av_t[2]], c='r', linewidth=0.5)
        gca.plot([0, av_t[0]], [0, av_t[1]], [0, -av_t[2]], c='r', linewidth=0.5)
        ax.scatter3D(four[:, 0], four[:, 1], four[:, 2], c=four[:, 1], s=10, cmap='Greens', label='cluster 4', marker='o')
        gca.plot([0, av_fo[0]], [0, av_fo[1]], [0, av_fo[2]], c='r', linewidth=0.5)
        gca.plot([0, -av_fo[0]], [0, -av_fo[1]], [0, -av_fo[2]], c='r', linewidth=0.5)
        ax.legend(bbox_to_anchor=(0.15, 0.95), markerscale=2.3, fontsize=13)
        # ax.legend(bbox_to_anchor=(0.81 / 0.83, 1.03), markerscale=1.3, fontsize=13)

    leg = ax.get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontweight='bold')
    plt.show()
    if save:
        fig.savefig('./result/fig/{}_fig.jpg'.format(dataset), dpi=200)


def get_ellipse(e_x, e_y, a, b, e_angle):
    """
    Drawing ellipse
    :param e_x: x-coordinate
    :param e_y: y-coordinate
    :param a: length of semimajor axis
    :param b: length of semiminor axis
    :param e_angle: angle between major axis and X axis of ellipse
    :return: None
    """
    angles_circle = np.arange(0, 2 * np.pi, 0.01)
    x = []
    y = []
    for angles in angles_circle:
        or_x = a * cos(angles)
        or_y = b * sin(angles)
        length_or = sqrt(or_x * or_x + or_y * or_y)
        or_theta = atan2(or_y, or_x)
        new_theta = or_theta + e_angle/180*np.pi
        new_x = e_x + length_or * cos(new_theta)
        new_y = e_y + length_or * sin(new_theta)
        x.append(new_x)
        y.append(new_y)
    return x, y


def plot_eplipse_result(root='./result/cdp-wmm', filename='nyu2_cdp.png'):

    plt.axis('off')
    img = plt.imread('{}/{}'.format(root, filename))
    plt.imshow(img)
    x, y = get_ellipse(47, 149, 35, 8, 0)
    plt.plot(x, y, c='w')
    x, y = get_ellipse(120, 150, 35, 8, 5)
    plt.plot(x, y, c='w')
    x, y = get_ellipse(240, 168, 35, 10, 5)
    plt.plot(x, y, c='w')

    plt.savefig('{}/mark_{}'.format(root, filename), bbox_inches='tight', dpi=200)
    # plt.show()


def change_png2eps(root='./result/figures'):

    files = file_name(root)[0]
    for index, name in enumerate(files):
        img = plt.imread('{}/{}'.format(root, name))
        f_index = name.find('.', 0)
        f_name = name[:f_index]
        plt.imsave('{}/test_{}.eps'.format(root, f_name), img, format='eps', dpi=500)

# plot_number_cluster(save=True)
# plot_3d('syn_data2', save=True)
# plot_eplipse_result()
# change_png2eps()
