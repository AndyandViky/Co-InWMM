# -*- coding: utf-8 -*-
# @author: andy
# @contact: andy_viky@163.com
# @github: https://github.com/AndyandViky
# @csdn: https://blog.csdn.net/AndyViky
# @file: train_synthetic.py
# @time: 2020/1/13 15:17
# @desc: train_synthetic.py
try:
    import os
    import argparse
    import numpy as np
    import matplotlib.pyplot as plt

    from model import VIModel, CVIModel
    from scipy import io as scio
    from sklearn.cluster import KMeans
    from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
    from vmfmix.model import VIDP, CVIDP

    from config import DATA_PARAMS, DATASETS_DIR, LOG_DIR, SEG_DIR, RESULT_DIR
    from utils import console_log, scalar_data, file_name
    from plot import plot_seg

except ImportError as e:
    print(e)
    raise ImportError


class Trainer:

    def __init__(self, args):

        if int(args.algorithm_category) == 0:
            self.model = VIModel(args)
        else:
            self.model = CVIModel(args)

    def train(self, data):

        self.model.fit(data)


if __name__ == "__main__":

    # set args
    parser = argparse.ArgumentParser(prog='HIN-datas',
                                    description='Hierarchical Dirichlet process Mixture Models of datas Distributions')
    parser.add_argument('-c', '--algorithm_category', dest='algorithm_category', help='choose VIModel:0 or SVIModel:1',
                        default=1)
    parser.add_argument('-name', '--data_name', dest='data_name', help='data_name', default='nyu')
    parser.add_argument('-lp', '--load_params', dest='load_params', help='load_params', default=1)
    parser.add_argument('-verbose', '--verbose', dest='verbose', help='verbose', default=1)
    # hyper parameters
    parser.add_argument('-t', '--T', dest='T', help='truncation level T', default=10)
    parser.add_argument('-gamma', '--gamma', dest='gamma', help='stick gamma', default=1)
    parser.add_argument('-z', '--z', dest='z', help='zeta', default=0.05)
    parser.add_argument('-u', '--u', dest='u', help='u', default=1)
    parser.add_argument('-v', '--v', dest='v', help='v', default=0.01)

    parser.add_argument('-mth', '--mix_threshold', dest='mix_threshold', help='mix_threshold', default=0.01)
    parser.add_argument('-m', '--max_iter', dest='max_iter', help='max iteration of variational inference', default=100)
    parser.add_argument('-sc', '--scalar', dest='scalar', help='data scalar', default=2)
    args = parser.parse_args()

    print('begin training......')
    print('========================dataset is {}========================'.format(args.data_name))

    # logger = open(os.path.join(LOG_DIR, "log.txt"), 'a')
    # logger.write(
    #     'begin training: ========================dataset is {}========================\n'.format(args.data_name)
    # )
    # logger.close()

    T, mix_threshold, algorithm_category, max_iter, dim, max_hy1f1_iter, gamma, z, u, v = DATA_PARAMS[
        args.data_name][args.algorithm_category]

    # pred = KMeans(n_clusters=20, max_iter=300).fit_predict(train_data)
    # category = np.unique(np.array(pred))
    # print(category)
    # console_log(pred[:2000], data=train_data[:2000], labels=None, model_name='===========kmeans')

    if int(args.load_params) == 1:
        args.T = T
        args.mix_threshold = mix_threshold
        args.max_iter = max_iter
        args.max_hy1f1_iter = max_hy1f1_iter
        args.gamma = gamma
        args.z = z
        args.u = u
        args.v = v

    # files = file_name(SEG_DIR)[0]
    # for index, name in enumerate(files):
    #     data = scio.loadmat('{}/{}'.format(SEG_DIR, name))
    #     nor_data = data['imgNormals']
    #
    #     train_data, size = scalar_data(nor_data, args.scalar)
    #
    #     trainer = Trainer(args)
    #     trainer.train(train_data)
    #     pred = trainer.model.predict(train_data)
    #     category = np.unique(np.array(pred))
    #     logger = open(os.path.join(LOG_DIR, "log.txt"), 'a')
    #     logger.write(
    #         'nyu{}: cluster: {}\n'.format(index+1, category)
    #     )
    #     logger.close()
    #     plot_seg(train_data, pred, size, nor_data=nor_data, file_name='nyu{}'.format(index+1), save=True)
    #     print(category)
    #     console_log(pred[:2000], data=train_data[:2000], labels=None, model_name='===========dp-wmm', newJ=len(category))
    data = scio.loadmat('{}/nyu0002.mat'.format(SEG_DIR))
    nor_data = data['imgNormals']

    # data = scio.loadmat('./datas/segmentation/nyu1.mat')
    # nor_data = data['rgbd_data'][0]['imgNormals'][0]

    train_data, size = scalar_data(nor_data, args.scalar)

    # pred = VIDP(n_cluster=3, max_iter=100).fit_predict(train_data)
    # category = np.unique(np.array(pred))
    # print(category)
    # plot_seg(train_data, pred, size, root='{}/wmm'.format(RESULT_DIR), file_name='nyu1215', save=False)

    trainer = Trainer(args)
    trainer.train(train_data)
    pred = trainer.model.predict(train_data)
    category = np.unique(np.array(pred))
    print(category)
    if algorithm_category == 1:
        RESULT_DIR = os.path.join(RESULT_DIR, 'cdp-wmm')
    elif algorithm_category == 0:
        RESULT_DIR = os.path.join(RESULT_DIR, 'dp-wmm')
    plot_seg(train_data, pred, size, root=RESULT_DIR, file_name='nyu0513', save=False)

