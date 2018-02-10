#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Jiajun Zhu


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import data_manager
import constants
import updater
import networks
import chainer
from chainer import Variable, optimizers, training
from chainer.training import extensions


def initialize_trainer(x_train, y_train, x_cv,  y_cv):
    """
    trainer を作成
    """
    train = list(zip(x_train, y_train))
    cv  = list(zip(x_cv,  y_cv))

    # 再現性確保
    np.random.seed(1)

    # モデルの宣言
    model = networks.LSTM(constants.SEQ_LENGTH, 100, 1)

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
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss']))
    return trainer, model


if __name__ == '__main__':

    # データを取得
    df = data_manager.return_chart_data(constants.PAIR_USDT_BTC, constants.PERIOD, constants.DAY)

    time_train, time_cv, time_test, x_train, y_train, x_cv, y_cv, x_test, y_test_not_normalized, y_scaler = data_manager.prepare_data(df)

    trainer, model = initialize_trainer(x_train, y_train, x_cv, y_cv)
    trainer.run()

    model.reset_state()
    y_pred_test = model.predict(Variable(x_test)).data

    plt.figure(figsize=(20,10))
    plt.plot(time_test, y_test_not_normalized, color='#2980b9', label = "real") # 実測値は青色
    plt.plot(time_test, y_scaler.inverse_transform(y_pred_test), color='#f39c12', label = "predicted") # 予測値はオレンジ
    plt.legend(loc='best', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.show()
