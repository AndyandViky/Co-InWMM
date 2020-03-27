# -*- coding: utf-8 -*-
# @author: andy
# @contact: andy_viky@163.com
# @github: https://github.com/AndyandViky
# @csdn: https://blog.csdn.net/AndyViky
# @file: train.py
# @time: 2020/1/13 15:17
# @desc: train.py
try:
    import argparse
    import numpy as np

    from model import VIModel
    from scipy import io as scio

    from config import DATA_PARAMS, DATASETS_DIR
    from utils import console_log, get_data

except ImportError as e:
    print(e)
    raise ImportError


class Trainer:

    def __init__(self, args):

        if int(args.algorithm_category) == 0:
            self.model = VIModel(args)
        else:
            pass

    def train(self, data):

        self.model.fit(data)


if __name__ == "__main__":

    # set args
    parser = argparse.ArgumentParser(prog='HIN-datas',
                                    description='Hierarchical Dirichlet process Mixture Models of datas Distributions')
    parser.add_argument('-c', '--algorithm_category', dest='algorithm_category', help='choose VIModel:0 or SVIModel:1',
                        default=0)
    parser.add_argument('-name', '--data_name', dest='data_name', help='data_name', default='big_data')
    parser.add_argument('-lp', '--load_params', dest='load_params', help='load_params', default=1)
    parser.add_argument('-verbose', '--verbose', dest='verbose', help='verbose', default=1)
    # hyper parameters
    parser.add_argument('-t', '--T', dest='T', help='truncation level T', default=10)
    parser.add_argument('-gamma', '--gamma', dest='gamma', help='second stick gamma', default=1)
    parser.add_argument('-mth', '--mix_threshold', dest='mix_threshold', help='mix_threshold', default=0.05)
    parser.add_argument('-m', '--max_iter', dest='max_iter', help='max iteration of variational inference', default=100)
    args = parser.parse_args()

    data = scio.loadmat('./datas/{}.mat'.format(args.data_name))
    labels = data['z'].reshape(-1).astype(np.int)
    data = data['data']

    # data1 = scio.loadmat('./datas/Data4.mat')['Data']
    # scio.savemat('./datas/big_data1.mat', {'data': data1, 'z': labels})
    print('begin training......')
    print('========================dataset is {}========================'.format(args.data_name))

    T, mix_threshold, algorithm_category, max_iter, dim = DATA_PARAMS[
        args.data_name]

    if int(args.load_params) == 1:
        args.T = T
        args.mix_threshold = mix_threshold
        args.algorithm_category = algorithm_category
        args.max_iter = max_iter

    trainer = Trainer(args)
    trainer.train(data)
    pred = trainer.model.predict(data)
    category = np.unique(np.array(pred))
    print(category)
    console_log(pred, labels=labels, model_name='===========hdp-vmf')

