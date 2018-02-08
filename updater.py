#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Jiajun Zhu


import chainer
from chainer import training

class LSTMUpdater(training.StandardUpdater):
    def __init__(self, data_iter, optimizer, device=None):
        super(LSTMUpdater, self).__init__(data_iter, optimizer, device=None)
        self.device = device

    def update_core(self):
        data_iter = self.get_iterator("main")
        optimizer = self.get_optimizer("main")

        batch = data_iter.__next__()
        x_batch, y_batch = chainer.dataset.concat_examples(batch, self.device)

        # ここで reset_state() を実行
        optimizer.target.reset_state()

        optimizer.target.cleargrads()
        loss = optimizer.target(x_batch, y_batch)
        loss.backward()
        # batch単位で古い情報を削除し、計算コストを削減
        loss.unchain_backward()
        # batch単位で更新
        optimizer.update()
