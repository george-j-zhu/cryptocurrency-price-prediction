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


class DataManager:

    def __init__(self):
        """
        constructor
        Args:

        """
        self.dataframe = None
        self.time_train = None
        self.time_test = None
        self.train_original_df = None
        self.test_original_df = None
        self.train_diffrenced = None
        self.test_diffrenced = None
        self.train_scaled = None
        self.test_scaled = None
        self.scaler = None
        self.train_percent = 0.8
        self.diff_interval = 1
        self.column = "close"


    def return_chart_data(self, pair, period, start_time, end_time):
        """
        retrieve data from poloniex.
        """
        polo = poloniex.Poloniex()
        chart_data = polo.returnChartData(pair,
                                          period=period,
                                          start=time.mktime(time.strptime(start_time, "%Y-%m-%d %H:%M")),
                                          end=time.mktime(time.strptime(end_time, "%Y-%m-%d %H:%M")))
        df = pd.DataFrame(chart_data)
        df["datetime"] = pd.to_datetime(df["date"].astype(int) , unit="s")
        df = df.set_index("datetime")

        fig, ax1 = plt.subplots(figsize=(20, 10))

        ax1.plot(df.index, df[self.column].astype(np.float32), label = "Coin Price", color="deeppink")

        #ax2 = ax1.twinx()
        #ax2.plot(df.index, df["quoteVolume"].astype(np.float32), label = "Trading Volume of Coin", color="dodgerblue")
        ax1.legend(loc=2, fontsize=14)
        #ax2.legend(loc=1, fontsize=14)
        ax1.tick_params(labelsize=14)
        #ax2.tick_params(labelsize=14)

        plt.show()

        self.dataframe = df.loc[:, ["close"]].astype(np.float32)


    def range_scale_data(self, matrix):
        """
        scale data to a specified range

        Args:
            dataframe: input data

        Returns:
            range scaled dataframe
        """
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        return min_max_scaler.fit_transform(matrix), min_max_scaler


    def inverse_scaled_data(self, data):
        """
        inverse scaled data
        """
        return self.scaler.inverse_transform(data)


    def __generate_series_data_for_supervised_learning(self, data_array, datetime_array):
        """
        split "close" column as explanatory_and_target_variables.
        """
        t = np.empty((0,1), int)
        x = np.empty((0,constants.SEQ_LENGTH), np.float32)
        y = np.empty((0,1), np.float32)

        m = len(data_array)
        for n in range(constants.SEQ_LENGTH, m):
            new_t = np.array([[datetime_array[n]]])
            new_x = np.array([data_array[n-constants.SEQ_LENGTH: n]])
            new_y = np.array([[data_array[n]]])
            t = np.append(t, new_t, axis=0) # append row
            x = np.append(x, new_x, axis=0)
            y = np.append(y, new_y, axis=0)

        self.dataframe = self.dataframe.iloc[constants.SEQ_LENGTH:, :]
        return t, np.concatenate([x, y], axis=1)


    def __difference_data(self, data_array):
        """
        difference the dataframe.
        this step trys to remove the trend.
        differncing interval is 1 here.
        """
        diff = np.empty((0,1), np.float32)
        for i in range(self.diff_interval, len(data_array)):
            value = data_array[i] - data_array[i - self.diff_interval]
            diff = np.append(diff, value)

        self.dataframe = self.dataframe.iloc[self.diff_interval:, :]
        return diff


    def inverse_differenced_data(self, y, previous_y):
        """
        this method only revert data for one step.
        """
        return y + previous_y


    def __split_data(self, data_array, datetime_array):
        """
        split dataset as training and test data.
        """
        m = len(data_array)

        train_batches = int(m * self.train_percent / constants.BATCH_SIZE)

        # 80% training data(cv included), 20% test data
        m_train = train_batches * constants.BATCH_SIZE

        time_train, time_test = datetime_array[:m_train], datetime_array[m_train:]
        data_train, data_test = data_array[:m_train], data_array[m_train:]

        self.train_original_df = self.dataframe.iloc[:m_train, :]
        self.test_original_df = self.dataframe.iloc[m_train:, :]

        self.time_train = time_train
        self.time_test = time_test

        self.train_diffrenced = data_train
        self.test_diffrenced = data_test


    def prepare_data(self):
        """
        prepare training, cross validation and test data.
        """
        # difference data
        data_array = self.__difference_data(self.dataframe[self.column].values)

        time_array, data_array = self.__generate_series_data_for_supervised_learning(data_array, self.dataframe.index)

        self.__split_data(data_array, time_array)

        data_train_scaled, scaler = self.range_scale_data(self.train_diffrenced)

        self.scaler = scaler
        self.train_scaled = data_train_scaled
        self.test_scaled = scaler.transform(self.test_diffrenced)


    def inverse_data(self, x, y_pred, previos_y):

        # inverse range
        y_pred_unscaled = self.inverse_scaled_data(np.concatenate([x, y_pred], axis=1))[:, -1]
        # inverse difference
        y_pred_indiffereced = self.inverse_differenced_data(y_pred_unscaled, previos_y)

        return y_pred_indiffereced
