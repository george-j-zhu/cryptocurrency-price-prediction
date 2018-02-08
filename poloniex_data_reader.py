#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Jiajun Zhu


import poloniex
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

polo = poloniex.Poloniex()

def return_chart_data(pair, period, day):

    # データを取得
    chart_data = polo.returnChartData(pair, period=period, start=time.time()-polo.DAY*day, end=time.time())
    df = pd.DataFrame(chart_data)

    plt.figure(figsize=(20,10), dpi=200)
    plt.plot(pd.to_datetime(df["date"].astype(int) , unit="s"), df["close"].astype(np.float32), label = "Coin Price")
    plt.plot(pd.to_datetime(df["date"].astype(int) , unit="s"), df["quoteVolume"].astype(np.float32), label = "BTC Volume")
    #plt.yticks(500, 10000, 10)
    #plt.ylim(10000, 12000)
    plt.legend(loc='best', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.show()

    # TODO dfの索引を時間にする
    return df
