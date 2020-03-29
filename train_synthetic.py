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
    from sklearn.cluster import KMeans

    from config import DATA_PARAMS, DATASETS_DIR
    from utils import console_log

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
    parser.add_argument('-name', '--data_name', dest='data_name', help='data_name', default='big_data4')
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
    args = parser.parse_args()

    data = scio.loadmat('./datas/{}.mat'.format(args.data_name))
    labels = data['z'].reshape(-1).astype(np.int)
    data = data['data']

    # labels = np.empty(6000)
    # labels[:1000] = 1
    # labels[1000:2000] = 2
    # labels[2000:3000] = 3
    # labels[3000:4000] = 4
    # labels[4000:5000] = 5
    # labels[5000:6000] = 6
    # labels = labels.astype(np.int)
    # data1 = scio.loadmat('./datas/Data5.mat')['Data']
    # scio.savemat('./datas/big_data4.mat', {'data': data1, 'z': labels})
    print('begin training......')
    print('========================dataset is {}========================'.format(args.data_name))

    T, mix_threshold, algorithm_category, max_iter, dim, max_hy1f1_iter, gamma, z, u, v = DATA_PARAMS[
        args.data_name][args.algorithm_category]

    if int(args.load_params) == 1:
        args.T = T
        args.mix_threshold = mix_threshold
        args.max_iter = max_iter
        args.max_hy1f1_iter = max_hy1f1_iter
        args.gamma = gamma
        args.z = z
        args.u = u
        args.v = v

    trainer = Trainer(args)
    trainer.train(data)
    pred = trainer.model.predict(data)
    category = np.unique(np.array(pred))
    print(category)
    console_log(pred, labels=labels, model_name='===========dp-wmm')

