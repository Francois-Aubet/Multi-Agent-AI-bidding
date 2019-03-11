from algorithms.CEAlgorithm import CEAlgorithm
from algorithms.ConstantBid import ConstantBid, ConstantBidCE
from algorithms.RandomBid import RandomBid
from algorithms.LinearBid import LinearBid, NonLinearBid
from algorithms.DictReader import DictReader
from algorithms.PRUD import PRUD
from Evaluator import Evaluator
from MultiEvalualOnline import MultiEvaluatorOnline
from MultiEvaluatorOff import MultiEvaluatorOff
from DataHandler import DataHandler
from algorithms.BidStream import BidStream

from models.CTRLogistic import CTRLogistic
from models.CTRXGboost import CTRXGboost

#import pickle
import dill as pickle

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
debug_mode = True

# create a data hanlder instance:
data_handler = DataHandler(train_set_path, vali_set_path, test_set_path, debug_mode)


file_class = open('pretrain/prepared_CTR_XGboost', 'r+b')
ctr_pred = pickle.load(file_class)

file_class = open('pretrain/prepared_CTR_Logistic', 'r+b')
ctr_pred2 = pickle.load(file_class)

file_class = open('pretrain/prepared_cdf_logistic', 'r+b')
pre_dict = pickle.load(file_class)

file_class = open('pretrain/prepared_cdf_xgb', 'r+b')
pre_dict2 = pickle.load(file_class)

file_class = open('pretrain/prepared_MP_XGboost', 'r+b')
pre_mp = pickle.load(file_class)

file_class = open('pretrain/prepared_ensemble', 'r+b')
pre_dict3 = pickle.load(file_class)



alg0 = BidStream(ctr_pred,2)

alg1 = LinearBid(ctr_pred_table = ctr_pred, base_params=145)
alg2 = LinearBid(ctr_pred_table = ctr_pred2, base_params=145)

alg3 = NonLinearBid(ctr_pred_table = ctr_pred)
alg4 = NonLinearBid(ctr_pred_table = ctr_pred2)

alg5 = DictReader(pre_dict) # uncertainty 1
alg6 = DictReader(pre_dict2) # uncertainty 2

alg7 = PRUD(ctr_pred, pre_mp)

alg8 = DictReader(pre_dict3) # ensemble

algo_list = [alg0, alg1, alg2, alg3, alg4, alg5, alg6, alg7, alg8, ConstantBid(), RandomBid()]

#,alg1, alg2, alg3, alg4, ConstantBid(), RandomBid()]
#RandomBid(),  BidStream(ctr_pred,2),


multi_eval = MultiEvaluatorOnline(algo_list,data_handler)
#multi_eval = MultiEvaluatorOff(algo_list,data_handler)

result, bid_list = multi_eval.evaluate()

#print(result)


bid_df = pd.DataFrame(bid_list, columns=['bidid','bidprice'])
bid_df.to_csv('bids2.csv', index=False)




list_budget_factor, list_base_bid, list_eff = alg0.get_meas_lists()
result = [list_budget_factor, list_base_bid, list_eff]
file_class = open('saving_the_meas3','w+b')
pickle.dump(result, file_class)






# alg5 = LinearBid(ctr_pred_table = ctr_pred, base_params=140)
# alg6 = LinearBid(ctr_pred_table = ctr_pred2, base_params=140)

# alg7 = NonLinearBid(ctr_pred_table = ctr_pred, base_params=(78,1e-6))
# alg8 = NonLinearBid(ctr_pred_table = ctr_pred2, base_params=(78,1e-6))

# fig, ax1 = plt.subplots()

# color = 'tab:blue'
# ax1.set_xlabel('Impressions')
# ax1.set_ylabel('budget factor', color=color)
# ax1.plot(range(len(list_budget_factor)), list_budget_factor, color=color)
# ax1.set_ylim((0.8,1.2))
# ax1.tick_params(axis='y', labelcolor=color)
# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

# color = 'tab:red'
# ax2.set_ylabel('mean base bid2', color=color)  # we already handled the x-label with ax1
# ax2.plot(range(len(list_base_bid)), list_base_bid, color=color)
# #ax2.plot(range(len(true_payprice)), true_payprice, color=color)
# ax2.tick_params(axis='y', labelcolor=color)

# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.show()













# instantiate algorithm
#ctrLog = CTRLogistic()
#ctrLog = CTRXGboost()
#ctrLog.train(data_handler)

#file_class = open('trained_CTRXGboost','w+b')
#file_class = open('trained_CTRLogistic','w+b')
#pickle.dump(ctrLog, file_class)

#file_class = open('trained_CTRXGboost','r+b')
#ctrLog = pickle.load(file_class)

# code to save the dictionary of predicted ctr:
# 
# ctr_train = ctrLog.predict(train_x)
# ctr_valid = ctrLog.predict(valid_x)
# ctr_test = ctrLog.predict(test_x)

# ctr_all = np.concatenate((ctr_train,ctr_valid,ctr_test))
# bidid_all = np.concatenate((train_x.bidid.values,valid_x.bidid.values,test_x.bidid.values))

# ctr_dict = dict(zip(bidid_all, ctr_all))

# file_class = open('prepared_CTR_XGboost','w+b')
# pickle.dump(ctr_dict, file_class)
#  