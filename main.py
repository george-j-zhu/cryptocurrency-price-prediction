#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Jiajun Zhu


import numpy as np
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
    initialize chianer trainer
    """
    train = list(zip(x_train, y_train))
    cv  = list(zip(x_cv,  y_cv))

    # initialize an LSTM
    model = networks.LSTM(constants.SEQ_LENGTH, 100, 1)

    # optimizer
    optimizer = optimizers.SGD()  #learning rate alpha=0.001?
    optimizer.setup(model)

    # batch
    batchsize = 20
    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    cv_iter = chainer.iterators.SerialIterator(cv, batchsize, repeat=False, shuffle=False)

    # updater
    upd = updater.LSTMUpdater(train_iter, optimizer)

    # trainer
    epoch = 30
    trainer = training.Trainer(upd, (epoch, 'epoch'))

    # extension
    trainer.extend(extensions.Evaluator(cv_iter, model))
    trainer.extend(extensions.LogReport(trigger=(1, 'epoch')))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss']))
    return trainer, model


def predict_y_test(x_seq, model, y_test):
    """
    predict one future value by a past sequence.
    """
    y_pred = np.zeros((len(y_test), 1))

    for n in range(len(y_test)):
        model.reset_state()
        _y_pred = model.predict(Variable(np.array([x_seq]))).data
        y_pred[n, 0] = _y_pred

        # create a new sequence for the next prediction
        x_seq = np.delete(x_seq, 0)
        x_seq = np.append(x_seq, _y_pred)

    return y_pred


if __name__ == '__main__':

    # fix the random generator
    np.random.seed(1)

    # read bitcoin data as dataframe
    df = data_manager.return_chart_data(constants.PAIR_USDT_BTC, constants.PERIOD, constants.DAY)

    # prepare dataset
    time_train, time_cv, time_test, x_train, x_cv, x_test, \
        y_train, y_cv, y_test_not_normalized, y_scaler = data_manager.prepare_data(df)

    # initialize trainer
    trainer, model = initialize_trainer(x_train, y_train, x_cv, y_cv)
    # train
    trainer.run()

    # reset states in our LSTM network
    model.reset_state()

    # make predictions for cross-validation set
    y_pred_cv = model.predict(Variable(x_cv)).data
    # plot predicitions
    plt.figure(figsize=(20,10))
    plt.plot(time_cv, y_scaler.inverse_transform(y_cv), label = "real")
    plt.plot(time_cv, y_scaler.inverse_transform(y_pred_cv), label = "predicted")
    plt.legend(loc='best', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.show()

    # make predictions for test set
    y_pred_test = predict_y_test(x_cv[-1], model, y_test_not_normalized)
    plt.figure(figsize=(20,10))
    # plot predictions
    plt.plot(time_test, y_test_not_normalized, label = "real")
    plt.plot(time_test, y_scaler.inverse_transform(y_pred_test), label = "predicted")
    plt.legend(loc='best', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.show()

    plt.figure(figsize=(20,10))
    # plot predictions
    plt.plot(time_test[:50], y_test_not_normalized[:50], label = "real")
    plt.plot(time_test[:50], y_scaler.inverse_transform(y_pred_test)[:50], label = "predicted")
    plt.legend(loc='best', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.show()
