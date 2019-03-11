from algorithms.CEAlgorithm import CEAlgorithm
from algorithms.IAlgorithm import IAlgorithm
import numpy as np


class RandomBid(IAlgorithm):
    """ The simplest constant bidding algorithm.

    TBD: this is just a test version designed to test the evaluation pipeline.
    """

    alg_name = "random"

    def __init__(self):
        """ Constructor of the abstract class.
            Should be called when initializing the child classes.
        """

        self._mean_bid = 77.14
        self._std_dib = 11.73

    def train(self, data_handler):
        """ Training the algorithm using self._train* .
        This needs to be overwritten in every child classes.

        @:return: the train error

        use tqdm to display the progress because dataset is big

        """
        pass

    def predict(self, test_x):
        """ Predicting for a dataset. This needs to be overwritten in every child classes.
        @:return: the bid
        """

        return np.random.randint(self._mean_bid - self._std_dib, self._mean_bid + self._std_dib, size=test_x.shape[0])

    def predict_single(self, impression, metric):
        """ Predicting for a single impression. T
        """

        return np.random.randint(self._mean_bid - self._std_dib, self._mean_bid + self._std_dib)


class RandomBidCE(IAlgorithm):
    """ The simplest constant bidding algorithm.

    TBD: this is just a test version designed to test the evaluation pipeline.
    """

    alg_name = "randomCE"

    def __init__(self, n_rounds, ceAlgo: CEAlgorithm):
        """ Constructor of the abstract class.
            Should be called when initializing the child classes.
        """

        self._mean_bid = 77
        self._std_dib = 11.7
        self.n_rounds = n_rounds
        self.ceAlgo = ceAlgo

    def train(self, data_handler):
        """ Training the algorithm using self._train* .
        This needs to be overwritten in every child classes.

        """
        _, train_y, _, valid_y, _ = data_handler.get_datasets()

        good_dists = np.zeros(self.n_rounds)
        train_y = train_y.assign(multipliers=1 * self._mean_bid)
        for i in range(self.n_rounds):
            df_y = train_y.sample(valid_y.shape[0])
            good_dist = self.ceAlgo.train(train_y, valid_y, is_random_case=True)
            good_dists[i] = good_dist

        self._base_bid = good_dists.mean()

        print(good_dists)

        print("Optimal base_bid: {}".format(self._base_bid))

    def predict(self, test_x):
        """ Predicting for a dataset. This needs to be overwritten in every child classes.
        @:return: the bid
        """

        return np.random.randint(self._mean_bid - self._std_dib, self._mean_bid + self._std_dib, size=test_x.shape[0])

    def predict_single(self, impression, metric):
        """ Predicting for a single impression. T
        """

        return np.random.randint(self._mean_bid - self._std_dib, self._mean_bid + self._std_dib)
