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
import scipy.io as scio

from matplotlib.pyplot import MultipleLocator

from config import RESULT_DIR, LOG_DIR


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
        if len(category) == 2:
            print(1)
        categorys.append(len(category))
    logger.close()

    categorys = np.array(categorys)

    ca_num = np.array([categorys[categorys == i].shape[0] for i in range(0, 11)])

    plt.tick_params(axis='both', which='major', labelsize=14)
    x_major_locator = MultipleLocator(1)
    y_major_locator = MultipleLocator(100)
    ax = plt.gca()
    ax.plot(ca_num, '-s', ms=6, alpha=1, mfc='blue', label='CDP-WMM')
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.xlim(-0.5, 10.5)

    plt.xlabel('Number of components', fontsize=14)
    plt.ylabel('Number of images', fontsize=14)

    plt.legend()

    if save:
        plt.savefig('{}/fig/category_fig.png'.format(RESULT_DIR), dpi=200)
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
        ax.legend(bbox_to_anchor=(0.14, 1), markerscale=1.5)

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
        ax.legend(bbox_to_anchor=(0.14, 0.95), markerscale=1.5)

        # ax.legend(bbox_to_anchor=(0.81 / 0.83, 1.03), markerscale=1.3, fontsize=13)

    plt.show()
    if save:
        fig.savefig('./result/{}_fig.jpg'.format(dataset), dpi=200)

# plot_number_cluster(save=False)
# plot_3d('syn_data1', save=True)


# 2: 33 42 66 75 83 145 162 164 168 177 184 192 197 217 448 513_ 558 593 657 765_ 797 1048 1202 1299 1307
# 2_: 513_

# 6: 22 90 94 106 114 227 228 241 321 403 810 859 1213 1225 1241 1243 1245_ 1334_ 1340 1345 1361 1426 1429 1449
# 6: 1334_

# 3: 1_ 2_ 18 69 140 143_ 153 202 252 342 385_ 409 410 450 457 462 463_ 486 499 557 568 619 662 680 726 846 878
# 916 928 941 963_ 983 1039 1047 1068 1094 1110 1119_ 1179 1185 1198 1199 1282 1314 1367 1420

# 4: 74 76 179_ 180_ 211 281 464 500 501_ 508 611 729 1084 1118_ 1163 1168_ 1200_ 1264 1317 1318

# 5: 1215

