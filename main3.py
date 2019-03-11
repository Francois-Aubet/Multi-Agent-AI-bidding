from sklearn.metrics import roc_auc_score, log_loss

from models.CTRLogistic import CTRLogistic
from models.CTRXGboost import CTRXGboost
from algorithms.CEAlgorithm import CEAlgorithm
from algorithms.LinearBid import NonLinearBid, LinearBid, LinearBidBF
from Evaluator import Evaluator
from DataHandler import DataHandler
import matplotlib.pyplot as plt

# just change the pandas printing options
import pandas as pd

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 1000)

# define the places of the datasets:

train_set_path = '../dataset/we_data/train.csv'
vali_set_path = '../dataset/we_data/validation.csv'
test_set_path = '../dataset/we_data/test.csv'

debug_mode = False

# create a data hanlder instance:
data_handler = DataHandler(train_set_path, vali_set_path, test_set_path, debug_mode)

# instantiate algorithm
# ctrLog1 = CTRLogistic()
ctrLog2 = CTRXGboost()
ceAlgo = CEAlgorithm(n_samples=100, p=0.3, max_iter=10, n_jobs=10)

# algo1 = LinearBid(ceAlgo, ctrModel=ctrLog1, n_rounds=1)
# algo2 = LinearBid(ceAlgo, ctrModel=ctrLog2, n_rounds=1)
algo2 = NonLinearBid(ctrModel=ctrLog2, use_pretrained=True)
algo2._base_params = (40, 1.2000000000000004e-06)

# evalua1 = Evaluator(algo1, data_handler)
evalua2 = Evaluator(algo2, data_handler)

# number_clicks1 = evalua1.evaluate()
# print(number_clicks1)
number_clicks2 = evalua2.evaluate()
print(number_clicks2)

_, _, vx, vy, _ = data_handler.get_datasets()
pred = algo2.ctrModel.predict(vx)
# pred2 = algo1.ctrModel.predict(vx)
roc_auc_score(vy.click, pred)
# roc_auc_score(vy.click, pred2)
log_loss(vy.click, pred)
# log_loss(vy.click, pred2)

bids = algo2.predict(vx)
plt.hist(bids, bins=200)
plt.show()
