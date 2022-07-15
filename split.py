# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 10:47:40 2022

@author: Admin
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 
from sklearn import model_selection
from sklearn.model_selection import RepeatedKFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
import os
def splitspectrum(interval_num, x_train, x_test):
    """
    :param interval_num:  int (common values are 10, 20, 30 or 40)
    :param x_train:  shape (n_samples, n_features)
    :param x_test:  shape (n_samples, n_features)
    :return: x_train_block:intervals splitting for training sets（dict）
            x_test_black： intervals splitting for test sets （dict）
    """
    feature_num = x_train.shape[1]
 
    x_train_block = {}
    x_test_black = {}
    remaining = feature_num % interval_num  # 用于检查是否能等分
    # （一）特征数量能够等分的情况
    if not remaining:
        interval_size = feature_num / interval_num  # 子区间波点数量
        for i in range(1, interval_num+1):
            # （1）取对应子区间的光谱数据
            feature_start, feature_end = int((i-1) * interval_size), int(i * interval_size)
            x_train_block[str(i)] = x_train[:, feature_start:feature_end]
            x_test_black[str(i)] = x_test[:, feature_start:feature_end]
 
    # （二）特征数量不能等分的情况(将多余波点等分到后面的几个区间里)
    else:
        separation = interval_num - remaining  # 前几个区间
        intervalsize1 = feature_num // interval_num
        intervalsize2 = feature_num // interval_num + 1
 
        # （2）前几个子区间(以separation为界)
        for i in range(1, separation+1):
            feature_start, feature_end = int((i-1) * intervalsize1), int(i * intervalsize1)
            x_train_block[str(i)] = x_train.loc[:, feature_start:feature_end]
            x_test_black[str(i)] = x_test.loc[:, feature_start:feature_end]
 
        # （3）后几个子区间(以separation为界)
        for i in range(separation+1, interval_num+1):
            feature_s = int((i - separation-1) * intervalsize2) + feature_end
            feature_e = int((i - separation) * intervalsize2) + feature_end
            x_train_block[str(i)] = x_train.loc[:, feature_s:feature_e]
            x_test_black[str(i)] = x_test.loc[:, feature_s:feature_e]
    return x_train_block, x_test_black