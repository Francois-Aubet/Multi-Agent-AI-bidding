from algorithms.CEAlgorithm import CEAlgorithm
from algorithms.IAlgorithm import IAlgorithm
import numpy as np


class DictReader(IAlgorithm):
    """ The simplest constant bidding algorithm.

    TBD: this is just a test version designed to test the evaluation pipeline.
    """

    alg_name = "DictReader"


    def __init__(self,dictionary):
        """ Constructor of the abstract class.
            Should be called when initializing the child classes.
        """

        self._dictionary = dictionary

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

        return self._dictionary[impression.bidid.values[0]]

