
from algorithms.CEAlgorithm import CEAlgorithm
from algorithms.ConstantBid import ConstantBid, ConstantBidCE
from algorithms.RandomBid import RandomBid, RandomBidCE
from Evaluator import Evaluator
from MultiEvalualOnline import MultiEvaluatorOnline
from MultiEvaluatorOff import MultiEvaluatorOff
from DataHandler import DataHandler

# just change the pandas printing options
import pandas as pd

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 1000)

# define the places of the datasets:

train_set_path = '../dataset/we_data/train.csv'
vali_set_path = '../dataset/we_data/validation.csv'
test_set_path = '../dataset/we_data/test.csv'


# this is just a thing to make the reding of the 
debug_mode = False

# create a data hanlder instance:
data_handler = DataHandler(train_set_path, vali_set_path, test_set_path, debug_mode)


if True:

    # instantiate algorithm
    ceAlgo = CEAlgorithm(n_samples=100, p=0.2, max_iter=10, n_jobs=10)
    ceBid = ConstantBidCE(n_rounds=3, ceAlgo=ceAlgo)

    evalua = Evaluator(ceBid, data_handler)

    number_clicks = evalua.evaluate()
    print(number_clicks)

else:  
    # instantiate algorithm
    ceAlgo = CEAlgorithm(n_samples=80, p=0.2, max_iter=10, n_jobs=10)
    ceBid = RandomBidCE(n_rounds=6, ceAlgo=ceAlgo)

    evalua = Evaluator(ceBid, data_handler)

    number_clicks = evalua.evaluate()

