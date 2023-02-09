# -*- coding: utf-8 -*-
# Author : JunHyuck Kim
# e-mail : junhyuck.kim@seculayer.com
# Powered by Seculayer © 2021 Service Model Team, R&D Center.
from tqdm import tqdm
import tensorflow as tf
import numpy as np

class EWCTrain(object):

    def __init__(self, optimizer, loss_fn, lambda_, prior_weights, model):
        self.optimizer =optimizer
        self.loss_fn = loss_fn
        self.lambda_ = lambda_
        self.prior_weights = prior_weights
        self.model = model


    def ewc_train(self, train_task, epochs, fisher_matrix, test_tasks):
        # empty list to collect per epoch test acc of each task
        for epoch in tqdm(range(epochs)):
            for batch in train_task:
                x, y = batch
                with tf.GradientTape() as tape:
                    pred = self.model(x)
                    loss = self.loss_fn(y, pred)
                    if fisher_matrix is not None:
                        loss += self._compute_penalty_loss(fisher_matrix)
                grads= tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        result = self._predict(test_tasks)
        return result

    def _compute_penalty_loss(self, fisher_matrix):
        penalty = 0.
        for u, v, w in zip(fisher_matrix, self.model.weights, self.prior_weights):
            penalty += tf.math.reduce_sum(u * tf.math.square(v - w))
        return 0.5 * self.lambda_ * penalty

    def _predict(self, test_tasks):
        pred = self.model.predict(test_tasks)
        return np.argmax(pred, 1)
