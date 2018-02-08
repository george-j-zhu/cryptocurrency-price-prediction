#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Jiajun Zhu


import chainer.links as L
import chainer.functions as F
from chainer import Chain, report


class LSTM(Chain):

    def __init__(self, n_input, n_units, n_output):
        """
        initialize a LSTM network
        """
        super(LSTM, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(n_input, n_units)
            self.l2 = L.LSTM(n_units, n_units)
            self.l3 = L.Linear(n_units, n_output)


    def __call__(self, x, t, train=True):
        """
        foward propagation
        """
        y = self.predict(x)
        loss = F.mean_squared_error(y, t)
        if train:
            report({'loss': loss}, self)
        return loss


    def predict(self, x):
        """
        make predictions
        """
        h1 = self.l1(x)
        h2 = self.l2(h1)
        y = self.l3(h2)
        return y


    def reset_state(self):
        """
        reset LSTM parameters
        """
        self.l2.reset_state()
