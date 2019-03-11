from abc import abstractmethod, ABCMeta


class CTRModel():
    """
    The abstract parent class for all the algorithms.
    """
    # we define the class as an abstract class:
    __metaclass__ = ABCMeta

    def __init__(self):
        """ Constructor of the abstract class.

            Should contain all the parameters of the algorithm.
        """

    @abstractmethod
    def train(self, data_handler):
        """ Training the algorithm using self._train* .
        This needs to be overwritten in every child classes.

        @:return: the train error

        use tqdm to display the progress because dataset is big

        """

        raise NotImplementedError("Must override methodB")

    @abstractmethod
    def predict(self, test_x):
        """ Predicting for a dataset. This needs to be overwritten in every child classes.
        @:return:
        """

        raise NotImplementedError("Must override methodB")
