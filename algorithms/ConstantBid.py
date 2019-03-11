from algorithms.CEAlgorithm import CEAlgorithm
from algorithms.IAlgorithm import IAlgorithm
import numpy as np


class ConstantBid(IAlgorithm):
    """ The simplest constant bidding algorithm.

    TBD: this is just a test version designed to test the evaluation pipeline.
    """

    alg_name = "constant"

    def __init__(self, base_bid=77.148):
        """ Constructor of the abstract class.
            Should be called when initializing the child classes.
        """

        self._base_bid = base_bid

    def train(self, data_handler):
        """ Training the algorithm using self._train* .
        This needs to be overwritten in every child classes.

        @:return: the train error

        use tqdm to display the progress because dataset is big

        """

        pass

    def predict(self, test_x, mode=None):
        """ Predicting for a dataset. This needs to be overwritten in every child classes.
        @:return: the bid
        """

        return np.repeat(self._base_bid, test_x.shape[0])

    def predict_single(self, impression, metric):
        """ Predicting for a single impression. T
        """

        return self._base_bid


class ConstantBidCE(IAlgorithm):
    alg_name = "constantCE"

    def __init__(self, n_rounds, ceAlgo: CEAlgorithm):
        """ Constructor of the abstract class.
            Should be called when initializing the child classes.
        """

        self._base_bid = None
        self.n_rounds = n_rounds
        self.ceAlgo = ceAlgo

    def train(self, data_handler):
        """ Training the algorithm using self._train* .
        This needs to be overwritten in every child classes.

        @:return: the train error

        use tqdm to display the progress because dataset is big.
        """
        _, train_y, _, valid_y, _ = data_handler.get_datasets()

        base_bids = np.zeros(self.n_rounds)
        train_y = train_y.assign(multipliers=1)
        for i in range(self.n_rounds):
            base_bid = self.ceAlgo.train(train_y, valid_y)
            base_bids[i] = base_bid

        self._base_bid = base_bids.mean()

        print("Optimal base_bid: {}".format(self._base_bid))

    def predict(self, test_x, mode=None):
        """ Predicting for a dataset. This needs to be overwritten in every child classes.
        @:return: the bid
        """

        return np.repeat(self._base_bid, test_x.shape[0])

    def predict_single(self, impression, metric):
        """ Predicting for a single impression. T
        """

        return self._base_bid
