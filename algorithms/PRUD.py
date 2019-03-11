import itertools
from abc import abstractmethod

import joblib

from Evaluator import _evaluate
from algorithms.CEAlgorithm import CEAlgorithm
from algorithms.IAlgorithm import IAlgorithm
import numpy as np
import pandas as pd


class PRUD(IAlgorithm):
    """ The simplest constant bidding algorithm.

    TBD: this is just a test version designed to test the evaluation pipeline.
    """

    alg_name = "PRUD"

    def __init__(self, ctr_pred, mp_pred, ceAlgo: CEAlgorithm = None, cutoff=np.exp(-11.89), n_rounds=1, submission=False):
        """ Constructor of the abstract class.
            Should be called when initializing the child classes.
        """

        self.mp_pred = mp_pred
        self.ctr_pred = ctr_pred
        self.n_rounds = n_rounds
        self.ceAlgo = ceAlgo
        self._cutoff = cutoff
        self.submission = submission

    def train_rho(self, data_handler):
        train_x, train_y, valid_x, valid_y, _ = data_handler.get_datasets()

        cutoff = np.zeros(self.n_rounds)

        if self.submission:
            rho = valid_x.bidid.apply(lambda x: self.ctr_pred[x] / self.mp_pred[x])
            pMP = valid_x.bidid.apply(lambda x: self.mp_pred[x])
            valid_y = valid_y.assign(rho=rho)
            valid_y = valid_y.assign(pMP=pMP)
            train_y = valid_y
        else:
            rho = train_x.bidid.apply(lambda x: self.ctr_pred[x] / self.mp_pred[x])
            pMP = train_x.bidid.apply(lambda x: self.mp_pred[x])
            train_y = train_y.assign(rho=rho)
            train_y = train_y.assign(pMP=pMP)

        for i in range(self.n_rounds):
            cutoff_val = self.ceAlgo.train(train_y, valid_y, is_prud_case=True)
            cutoff[i] = cutoff_val

        self._cutoff = cutoff.mean()

        print("Optimal cutoff: {}".format(self._cutoff))

    def train(self, data_handler):
        """ Training the algorithm using self._train* .
        This needs to be overwritten in every child classes.

        @:return: the train error

        use tqdm to display the progress because dataset is big

        """

        if self._cutoff is None:
            # rerun base bid if ctr model was trained
            self.train_rho(data_handler)

    def predict(self, test_x, mode='train'):
        """ Predicting for a dataset. This needs to be overwritten in every child classes.
        @:return: the bid
        """

        test_x['rho'] = test_x.bidid.apply(lambda x: self.ctr_pred[x] / self.mp_pred[x])
        test_x['pMP'] = test_x.bidid.apply(lambda x: self.mp_pred[x])

        bid = test_x.apply(lambda x: (x.pMP + 300) if x.rho > self._cutoff else 0.0, axis=1).values

        return bid

    def predict_single(self, impression, metric):
        """ Predicting for a dataset. This needs to be overwritten in every child classes.
        @:return: the bid
        """

        impression['rho'] = impression.bidid.apply(lambda x: self.ctr_pred[x] / self.mp_pred[x])
        impression['pMP'] = impression.bidid.apply(lambda x: self.mp_pred[x])

        bid = impression.apply(lambda x: (x.pMP + 300) if x.rho > self._cutoff else 0.0, axis=1).values

        return bid
