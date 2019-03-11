import itertools
from abc import abstractmethod

import joblib

from Evaluator import _evaluate
from algorithms.CEAlgorithm import CEAlgorithm
from algorithms.IAlgorithm import IAlgorithm
import numpy as np
import pandas as pd


class LinearBid(IAlgorithm):
    """ The simplest constant bidding algorithm.

    TBD: this is just a test version designed to test the evaluation pipeline.
    """

    alg_name = "linear"

    def __init__(self, ceAlgo: CEAlgorithm = None, ctrModel=None, n_rounds=None, base_params=None,
                 use_pretrained=False, submission=False, ctr_pred_table = None):
        """ Constructor of the abstract class.
            Should be called when initializing the child classes.
        """
        assert base_params is not None or (ceAlgo is not None and n_rounds is not None), \
            'You must provide either base_bid or ceAlgo'
        self._base_params = base_params
        self.ceAlgo = ceAlgo
        self.ctrModel = ctrModel
        self.n_rounds = n_rounds
        self.use_pretrained = use_pretrained
        self.submission = submission
        self._ctr_pred_table = ctr_pred_table

        if ctr_pred_table is not None:
            self._avg_ctr_all = np.array(list(ctr_pred_table.values())).mean()

    def train_base_bid(self, data_handler):
        train_x, train_y, _, valid_y, _ = data_handler.get_datasets()

        base_bids = np.zeros(self.n_rounds)

        if self.submission:
            multipliers = self.p_ctr_valid / self.avg_ctr_valid
            valid_y = valid_y.assign(multipliers=multipliers)
            train_y = valid_y
        else:
            multipliers = self.p_ctr_train / self.avg_ctr_train
            train_y = train_y.assign(multipliers=multipliers)

        for i in range(self.n_rounds):
            base_bid = self.ceAlgo.train(train_y, valid_y)
            base_bids[i] = base_bid

        self._base_params = base_bids.mean()

        print("Optimal base_bid: {}".format(self._base_params))

    def train(self, data_handler):
        """ Training the algorithm using self._train* .
        This needs to be overwritten in every child classes.

        @:return: the train error

        use tqdm to display the progress because dataset is big

        """
        train_x, train_y, valid_x, _, test_x = data_handler.get_datasets()

        train_model = False
        if not self.use_pretrained or self.ctrModel is None:
            self.ctrModel.train(data_handler, self.submission)
            train_model = True

        self.p_ctr_train = self.ctrModel.predict(train_x)
        self.p_ctr_valid = self.ctrModel.predict(valid_x)
        self.p_ctr_test = self.ctrModel.predict(test_x)

        self.avg_ctr_train = self.p_ctr_train.mean()
        self.avg_ctr_valid = self.p_ctr_valid.mean()
        self.avg_ctr_test = self.p_ctr_test.mean()

        if self._base_params is None or train_model:
            # rerun base bid if ctr model was trained
            self.train_base_bid(data_handler)

    def predict(self, test_x, mode='train'):
        """ Predicting for a dataset. This needs to be overwritten in every child classes.
        @:return: the bid
        """

        if mode == 'train':
            avg_ctr = self.avg_ctr_train
        elif mode == 'valid':
            avg_ctr = self.avg_ctr_valid
        elif mode == 'test':
            avg_ctr = self.avg_ctr_test
        else:
            avg_ctr = self.avg_ctr_train

        p_ctr = self.ctrModel.predict(test_x)

        return self._base_params * p_ctr / avg_ctr


    def predict_single(self,impression,metric):
        """ Predicting for a single impression. T
        """

        p_ctr = self._ctr_pred_table[impression.bidid.values[0]]

        return self._base_params * p_ctr / self._avg_ctr_all



class NonLinearBid(LinearBid):

    alg_name = "non-linear"

    def __init__(self, ctrModel=None, base_params=(40,1.2e-6), n_jobs=10, use_pretrained=False, ctr_pred_table = None, submission=False):
        self._base_params = base_params
        self.ctrModel = ctrModel
        self.use_pretrained = use_pretrained
        self.n_jobs = n_jobs
        self._ctr_pred_table = ctr_pred_table
        self.submission = submission

    def train_base_bid(self, data_handler):
        train_x, train_y, _, vali_y, _ = data_handler.get_datasets()
        c_params = [20, 40, 60, 80]
        lmd_params = np.linspace(1e-5, 1e-7, 100)

        all_params = list(itertools.product(c_params, lmd_params))

        sample_idx = np.random.choice(np.arange(train_x.shape[0]), vali_y.shape[0], replace=False)

        df_x = train_x.iloc[sample_idx]
        df_y = train_y.iloc[sample_idx]

        p_ctr = self.ctrModel.predict(df_x)

        results = joblib.Parallel(n_jobs=self.n_jobs, verbose=0)(
            joblib.delayed(_evaluate)(df_y, np.sqrt((c / lmd) * p_ctr + c ** 2) - c) for c, lmd in all_params
        )

        clicks = np.array([r[0] for r in results])

        c, lmd = all_params[clicks.argmax()]

        self._base_params = c, lmd

        print("Optimal c, lmd: {}, {}".format(c, lmd))

    def predict(self, test_x, mode=None):
        c, lmd = self._base_params
        p_ctr = self.ctrModel.predict(test_x)

        ret = np.sqrt((c / lmd) * p_ctr + c ** 2) - c

        return ret


    def predict_single(self,impression,metric):
        """ Predicting for a single impression. T
        """
        #c = 80
        #lmd = 1e-6
        c, lmd = self._base_params

        p_ctr = self._ctr_pred_table[impression.bidid.values[0]]

        ret = np.sqrt((c / lmd) * p_ctr + c ** 2) - c

        return ret



class LinearBidBF(LinearBid):
    """
    Linear Bidding with Brute Force Search for base_bid
    """

    def __init__(self, ctrModel=None, base_params=None, n_jobs=10, use_pretrained=False):
        self._base_params = base_params
        self.ctrModel = ctrModel
        self.use_pretrained = use_pretrained
        self.n_jobs = n_jobs

    def train_base_bid(self, data_handler):
        train_x, train_y, _, vali_y, _ = data_handler.get_datasets()
        lmd_params = np.linspace(10, 200, 300)

        all_params = lmd_params

        sample_idx = np.random.choice(np.arange(train_x.shape[0]), vali_y.shape[0], replace=False)

        df_x = train_x.iloc[sample_idx]
        df_y = train_y.iloc[sample_idx]

        p_ctr = self.ctrModel.predict(df_x)

        multipliers = p_ctr / self.avg_ctr

        results = joblib.Parallel(n_jobs=self.n_jobs, verbose=0)(
            joblib.delayed(_evaluate)(df_y, multipliers * base_bid) for base_bid in all_params
        )

        clicks = np.array([r[0] for r in results])

        self._base_params = all_params[clicks.argmax()]

        print("Optimal base_bid: {}".format(self._base_params))

    def predict(self, test_x):
        """ Predicting for a dataset. This needs to be overwritten in every child classes.
        @:return: the bid
        """

        p_ctr = self.ctrModel.predict(test_x)

        return self._base_params * p_ctr / self.avg_ctr
