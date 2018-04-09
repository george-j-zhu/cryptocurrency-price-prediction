#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Jiajun Zhu


# interval of each data (seconds）
PERIOD = 300

# currency pairs
PAIR_USDT_BTC = "USDT_BTC"
PAIR_BTC_ETH = "BTC_ETH"

# length for expanatory variable sequence(n features).
SEQ_LENGTH = 30 # aka. time steps in keras

BATCH_SIZE = 1 # as we need to use one-step forecast, batch size should be 1.
LSTM_HIDDEN_NEURONS = 32 # number of hidden nerons
DATA_DIM = 1
EPOCHS = 30
