#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Jiajun Zhu


import chainer
from chainer import training, Variable

class LSTMUpdater(training.StandardUpdater):
    def __init__(self, data_iter, optimizer, device=-1):
        super(LSTMUpdater, self).__init__(data_iter, optimizer, device=device)
        self.device = device

    def update_core(self):
        data_iter = self.get_iterator("main")
        optimizer = self.get_optimizer("main")

        batch = data_iter.__next__()
        x_batch, y_batch = chainer.dataset.concat_examples(batch, self.device)

        optimizer.target.cleargrads()
        # reset states in LSTM each update
        optimizer.target.reset_state()
        loss = optimizer.target(Variable(x_batch), Variable(y_batch))
        
        loss.backward()
        # delete history to reduce computation cost.
        loss.unchain_backward()
        # update per batch
        optimizer.update()
