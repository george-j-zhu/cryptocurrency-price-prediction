#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Jiajun Zhu

import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import poloniex_data_reader
import constants
import updater
import networks
import chainer
from chainer import Variable, optimizers, training
from chainer.training import extensions


def range_scale_data(matrix):
    """
    scale data to a specified range

    Args:
        dataframe: input data

    Returns:
        range scaled dataframe
    """
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    return min_max_scaler.fit_transform(matrix)

def generate_explanatory_and_target_variables(dataframe):
    """
    説明変数と目的変数を作成
    """
    data = dataframe['close'].astype(np.float32)

    # closeから説明変数と目的変数を作成する
    x, y = [], []
    N = len(data)
    M = 30 # 時系列の目的変数では、目的変数の一つの値に対してその直近の30サンプルの実測値を説明変数の値として使用する
    for n in range(M, N):
        # 入力変数と出力変数の切り分け
        _x = data[n-M: n] # 説明変数
        _y = data[n] # 目的変数
        # 計算用のリスト(x, y)に追加していく
        x.append(_x)
        y.append(_y)

    # numpyの形式に変換する（何かと便利なため）
    x = range_scale_data(np.array(x))
    y = np.array(y).reshape(len(y), 1)  # reshapeは後々のChainerでエラーが出ない対策

    # normalize y by dividing max(this can make it easier to inverse the output)
    norm_rate_for_y = y.max()
    y = y/norm_rate_for_y

    return x, y, norm_rate_for_y


def prepare_data(df):

    x, y, norm_rate_for_y = generate_explanatory_and_target_variables(df)

    N = len(x)
    # 70%を訓練用、30%を検証用
    N_train = int(N * 0.7)
    x_train, x_cv = x[:N_train], x[N_train:]
    y_train, y_cv = y[:N_train], y[N_train:]

    return x_train, y_train, x_cv, y_cv, norm_rate_for_y


def initialize_trainer(x_train, y_train, x_cv,  y_cv):
    """
    trainer を作成
    """
    train = list(zip(x_train, y_train))
    cv  = list(zip(x_cv,  y_cv))

    # 再現性確保
    np.random.seed(1)

    # モデルの宣言
    model = networks.LSTM(30, 100, 1)

    # optimizerの定義
    optimizer = optimizers.SGD()  #学習率を設定する alpha=0.001?
    optimizer.setup(model)

    # iteratorの定義
    batchsize = 20
    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    test_iter = chainer.iterators.SerialIterator(cv, batchsize, repeat=False, shuffle=False)

    # updaterの定義
    upd = updater.LSTMUpdater(train_iter, optimizer)

    # trainerの定義
    epoch = 30
    trainer = training.Trainer(upd, (epoch, 'epoch'))
    # trainerの拡張機能
    trainer.extend(extensions.Evaluator(test_iter, model)) # 評価データで評価
    trainer.extend(extensions.LogReport(trigger=(1, 'epoch'))) # 学習結果の途中を表示する
    # １エポックごとに、trainデータに対するlossと、testデータに対するlossを出力させる
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'elapsed_time']), trigger=(1, 'epoch'))
    return trainer, model


def execute_lstm():
    """
    lstm
    """
    # データを取得
    df = poloniex_data_reader.return_chart_data(constants.PAIR_USDT_BTC, constants.PERIOD, constants.DAY)
    #df = poloniex_data_reader.return_chart_data(constants.PAIR_BTC_ETH, constants.PERIOD, constants.DAY)

    x_train, y_train, x_cv, y_cv, norm_rate_for_y = prepare_data(df)

    trainer, model = initialize_trainer(x_train, y_train, x_cv, y_cv)
    trainer.run()

    # 予測値の計算
    model.reset_state()
    y_pred_train = model.predict(Variable(x_cv)).data

    # プロット
    plt.figure(figsize=(20,10), dpi=200)
    plt.plot(y_cv * norm_rate_for_y, color='#2980b9') # 実測値は青色
    plt.plot(y_pred_train * norm_rate_for_y, color='#f39c12') # 予測値はオレンジ
    plt.show()
