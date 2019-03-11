from abc import ABCMeta, abstractmethod
from Metrics import Metrics
import numpy as np
from tqdm import tqdm


class IAlgorithm():
    """
    The abstract parent class for all the algorithms.
    """
    # we define the class as an abstract class:
    __metaclass__ = ABCMeta

    alg_name = "not_named"

    def __init__(self):
        """ Constructor of the abstract class.

            Should contain all the parameters of the algorithm.
        """

    @abstractmethod
    def train(self):
        """ Training the algorithm using self._train* .
        This needs to be overwritten in every child classes.

        @:return: the train error

        use tqdm to display the progress because dataset is big

        """

        raise NotImplementedError("Must override methodB")

    @abstractmethod
    def predict(self):
        """ Predicting for a dataset. This needs to be overwritten in every child classes.
        @:return: 
        """

        raise NotImplementedError("Must override methodB")

    @abstractmethod
    def predict_single(self, impression, metric: Metrics):
        """ Predicting for a single impression. This needs to be overwritten in every child classes.
        @:return: 
        """

        raise NotImplementedError("Must override methodB")

        