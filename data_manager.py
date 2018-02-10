#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Jiajun Zhu


import poloniex
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
import constants

polo = poloniex.Poloniex()


def return_chart_data(pair, period, day):
    """
    retrieve data from poloniex.
    """
    chart_data = polo.returnChartData(pair, period=period, start=time.time()-polo.DAY*day, end=time.time())
    df = pd.DataFrame(chart_data)

    plt.figure(figsize=(20,10))
    plt.plot(pd.to_datetime(df["date"].astype(int) , unit="s"), df["close"].astype(np.float32), label = "Coin Price")
    #plt.plot(pd.to_datetime(df["date"].astype(int) , unit="s"), df["quoteVolume"].astype(np.float32), label = "Volume")
    #plt.yticks(500, 10000, 10)
    #plt.ylim(10000, 12000)
    plt.legend(loc='best', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.show()

    return df


def range_scale_data(matrix):
    """
    scale data to a specified range

    Args:
        dataframe: input data

    Returns:
        range scaled dataframe
    """
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    return min_max_scaler.fit_transform(matrix), min_max_scaler


def __generate_explanatory_and_target_variables(dataframe):
    """
    split close as explanatory_and_target_variables.
    """
    data = dataframe["close"].astype(np.float32)
    time = pd.to_datetime(dataframe["date"].astype(int) , unit="s")
    
    x = np.empty((0,constants.SEQ_LENGTH), np.float32)
    y = np.empty((0,1), np.float32)
    
    m = len(data)
    for n in range(constants.SEQ_LENGTH, m):
        new_x = np.array([data[n-constants.SEQ_LENGTH: n].as_matrix()])
        new_y = np.array([[data[n]]])
        x = np.append(x, new_x, axis=0)
        y = np.append(y, new_y, axis=0)
    return time, x, y


def prepare_data(dataframe):
    """
    prepare traning and validation data
    """
    time, x, y = __generate_explanatory_and_target_variables(dataframe)

    m = len(x)
    
    # 60% training data, 20% cv data, 20% test data
    N_train = int(m * 0.6)
    N_cv = int(m * 0.2)
    time_train, time_cv, time_test = time[:N_train], time[N_train:N_train+N_cv], time[N_train+N_cv:]
    x_train, x_cv, x_test = x[:N_train], x[N_train:N_train+N_cv], x[N_train+N_cv:]
    y_train, y_cv, y_test = y[:N_train], y[N_train:N_train+N_cv], y[N_train+N_cv:]
    
    x_train, null = range_scale_data(x_train)
    x_cv, null = range_scale_data(x_cv)
    x_test, null = range_scale_data(x_test)
    y_train, null = range_scale_data(y_train)
    y_cv, y_scaler = range_scale_data(y_cv)
    # in real production, since there will be no y_test, dont use scaler of y_test to inverse predictions.
    y_test_not_normalized = y_test
    
    return time_train, time_cv, time_test, x_train, y_train, x_cv, y_cv, x_test, y_test_not_normalized, y_scaler
