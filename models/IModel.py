from abc import abstractmethod, ABCMeta


class IModel():
    """
    The abstract parent class for all the models.
    """
    # we define the class as an abstract class:
    __metaclass__ = ABCMeta

    def __init__(self):
        """ Constructor of the abstract class.

            Should contain all the parameters of the models.
        """

    @abstractmethod
    def train(self, data_handler):
        """ Training the model using self._train* .
        This needs to be overwritten in every child classes.
        """

        raise NotImplementedError("Must override methodB")

    @abstractmethod
    def predict(self, test_x):
        """ Predicting for a dataset. This needs to be overwritten in every child classes.
        @:return:
        """

        raise NotImplementedError("Must override methodB")
