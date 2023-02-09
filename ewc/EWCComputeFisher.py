# -*- coding: utf-8 -*-
# Author : JunHyuck Kim
# e-mail : junhyuck.kim@seculayer.com
# Powered by Seculayer © 2021 Service Model Team, R&D Center.
import numpy as np
import tensorflow as tf

class EWCComputeFisher(object):

    def __init__(self, prior_model, data_samples, num_sample):
        self.prior_model = prior_model
        self.prior_weights = prior_model.weights
        self.num_sample = num_sample
        self.data_samples = data_samples
        self.fisher_matrix = self._compute_fisher()

    def _compute_fisher(self):
        weights = self.prior_weights
        fisher_accum = np.array([np.zeros(layer.numpy().shape) for layer in weights],
                                dtype=object
                                )
        for j in range(self.num_sample):
            idx = np.random.randint(self.data_samples.shape[0])
            with tf.GradientTape() as tape:
                logits = tf.nn.log_softmax(self.prior_model(np.array([self.data_samples[idx]])))
            grads = tape.gradient(logits, weights)
            for m in range(len(weights)):
                fisher_accum[m] += np.square(grads[m])
        fisher_accum /= self.num_sample
        return fisher_accum

    def get_fisher(self):
        return self.fisher_matrix