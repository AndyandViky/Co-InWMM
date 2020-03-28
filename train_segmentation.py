# -*- coding: utf-8 -*-
# @author: andy
# @contact: andy_viky@163.com
# @github: https://github.com/AndyandViky
# @csdn: https://blog.csdn.net/AndyViky
# @file: train_synthetic.py
# @time: 2020/1/13 15:17
# @desc: train_synthetic.py
try:
    import argparse
    import numpy as np

    from model import VIModel, CVIModel
    from scipy import io as scio

    from config import DATA_PARAMS, DATASETS_DIR
    from utils import console_log, scalar_data

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
                        default=0)
    parser.add_argument('-name', '--data_name', dest='data_name', help='data_name', default='nyu')
    parser.add_argument('-lp', '--load_params', dest='load_params', help='load_params', default=1)
    parser.add_argument('-verbose', '--verbose', dest='verbose', help='verbose', default=1)
    # hyper parameters
    parser.add_argument('-t', '--T', dest='T', help='truncation level T', default=10)
    parser.add_argument('-gamma', '--gamma', dest='gamma', help='second stick gamma', default=1)
    parser.add_argument('-mth', '--mix_threshold', dest='mix_threshold', help='mix_threshold', default=0.05)
    parser.add_argument('-m', '--max_iter', dest='max_iter', help='max iteration of variational inference', default=100)
    parser.add_argument('-sc', '--scalar', dest='scalar', help='data scalar', default=2)
    args = parser.parse_args()

    data = scio.loadmat('./datas/segmentation/{}1.mat'.format(args.data_name))
    data = data['rgbd_data']
    nor_data = data[0]['imgNormals'][0]
    rgb_data = data[0]['rgbImg'][0]
    dep_data = data[0]['depImg'][0]

    nor_data = scalar_data(nor_data, args.scalar)

    print('begin training......')
    print('========================dataset is {}========================'.format(args.data_name))

    T, mix_threshold, algorithm_category, max_iter, dim, max_hy1f1_iter = DATA_PARAMS[
        args.data_name]

    if int(args.load_params) == 1:
        args.T = T
        args.mix_threshold = mix_threshold
        args.algorithm_category = algorithm_category
        args.max_iter = max_iter
        args.max_hy1f1_iter = max_hy1f1_iter

    trainer = Trainer(args)
    trainer.train(nor_data)
    pred = trainer.model.predict(nor_data)
    category = np.unique(np.array(pred))
    print(category)
    console_log(pred[:2000], data=nor_data[:2000], labels=None, model_name='===========cdp-wmm')

