#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Jiajun Zhu


import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, Flatten
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import dataManager
import constants


# perform one-step forecast(with test data)
def one_step_prediction_using_test_data(model, previos_y):

    y_pred = np.zeros((len(dm.test_scaled), 1))

    for i in range(len(dm.test_scaled)):

        x = dm.test_scaled[i, :-1].reshape(constants.BATCH_SIZE, constants.SEQ_LENGTH, constants.DATA_DIM)
        _y_pred = model.predict(x)

        x = x.reshape(constants.BATCH_SIZE, constants.SEQ_LENGTH)
        _y_pred = _y_pred.reshape(constants.BATCH_SIZE, 1)

        # inverse scale
        y_pred_indiffereced = dm.inverse_data(x, _y_pred, previos_y)

        previos_y = dm.test_original_df.iloc[i, -1]

        model.reset_states()
        y_pred[i, 0] = y_pred_indiffereced

        # create a new sequence for the next prediction
        x = np.delete(x, 0)
        x = np.append(x, _y_pred[0, 0])

    return y_pred


if __name__ == '__main__':

    # fix the random generator
    np.random.seed(1)

    # coin pairs: USDT-Bitcoin
    dm = dataManager.DataManager()
    dm.return_chart_data(constants.PAIR_USDT_BTC, constants.PERIOD, "2018-03-30 00:00", "2018-04-01 00:00")
    dm.prepare_data()

    # definition of NN
    model = Sequential()
    model.add(LSTM(32, return_sequences=False, batch_input_shape=(constants.BATCH_SIZE, constants.SEQ_LENGTH, constants.DATA_DIM), stateful=True))
    #model.add(LSTM(32, stateful=True))
    model.add(Dense(1, activation='linear'))
    #model.add(Dropout(0.5))

    # learning rate as default
    model.compile(loss="mean_squared_error", optimizer="rmsprop")

    print(model.summary())

    # split CV set from training set.
    train_data_scaled, cv_data_scaled = train_test_split(
        dm.train_scaled, test_size=0.25, random_state=42, shuffle=False)
    x_train_scaled, y_train_scaled = train_data_scaled[:, 0:-1], train_data_scaled[:, -1]
    x_cv_scaled, y_cv_scaled = cv_data_scaled[:, 0:-1], cv_data_scaled[:, -1]
    # reshape data as a tensor(3 dims)
    x_train_scaled = x_train_scaled.reshape((x_train_scaled.shape[0], constants.SEQ_LENGTH, constants.DATA_DIM))
    print(x_train_scaled.shape)
    x_cv_scaled = x_cv_scaled.reshape((x_cv_scaled.shape[0], constants.SEQ_LENGTH, constants.DATA_DIM))
    # fit data
    model.fit(x_train_scaled, y_train_scaled,
            batch_size=constants.BATCH_SIZE,
            epochs=constants.EPOCHS,
            verbose=1,
            validation_data=(x_cv_scaled, y_cv_scaled),
            shuffle=False)


    predicted_test = one_step_prediction_using_test_data(model, dm.train_original_df.iloc[-1, -1])

    # plot predicitions
    plt.figure(figsize=(20,10))
    # get the original test data(not differenced)
    plt.plot(dm.time_test, dm.test_original_df.iloc[:, -1], label = "real")
    plt.plot(dm.time_test, predicted_test, label = "predicted")
    plt.legend(loc='best', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.show()