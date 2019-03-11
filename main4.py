import pickle

from models.CTRXGboost import CTRXGboost
from algorithms.PolicyGradient import PolicyGradient
from Evaluator import Evaluator
from MultiEvalualOnline import MultiEvaluatorOnline
from MultiEvaluatorOff import MultiEvaluatorOff
from DataHandler import DataHandler
from algorithms.BidStream import BidStream

import matplotlib.pyplot as plt
import numpy as np

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

file_class = open('pretrain/prepared_CTR_XGboost', 'r+b')
ctr_pred = pickle.load(file_class)

config = dict(LR_A=0.001, LR_C=0.01, A_LBOUND=0.01, A_UBOUND=10)
algo_list = [PolicyGradient(config, ctr_pred=ctr_pred)]

# multi_eval = MultiEvaluatorOnline([algo1,algo2],data_handler)
multi_eval = MultiEvaluatorOnline(algo_list, data_handler, use_pretrained=True)

result = multi_eval.evaluate(use_train=True)
