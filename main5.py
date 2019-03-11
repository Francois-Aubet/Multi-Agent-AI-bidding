import pickle

from Evaluator import Evaluator
from algorithms.CEAlgorithm import CEAlgorithm
from algorithms.PRUD import PRUD
from algorithms.PolicyGradient import PolicyGradient
from MultiEvalualOnline import MultiEvaluatorOnline
from DataHandler import DataHandler
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

file_class_ctr = open('pretrain/prepared_CTR_XGboost', 'r+b')
ctr_pred = pickle.load(file_class_ctr)
file_class_mp = open('pretrain/prepared_MP_XGboost', 'r+b')
mp_pred = pickle.load(file_class_mp)

# ctr_df = pd.DataFrame.from_dict(ctr_pred, orient='index')
# mp_df = pd.DataFrame.from_dict(mp_pred, orient='index')
#
# df = ctr_df.join(mp_df, lsuffix='cr', rsuffix='mp')
# df['rho'] = df['0cr']/df['0mp']
#
# np.log(df[df['rho'] > 0].rho).hist(bins=100)
# plt.show()
#
# np.std(np.log(df[df['rho'] > 0].rho))

ceAlgo = CEAlgorithm(n_samples=100, p=0.25, max_iter=10, n_jobs=10)
prud = PRUD(ceAlgo, ctr_pred, mp_pred, submission=True)

evalua = Evaluator(prud, data_handler)

number_clicks = evalua.evaluate()
print(number_clicks)

tx, ty, vx, vy, tex = data_handler.get_datasets()
bids = prud.predict(tex, mode='test')

submission = tex.bidid.to_frame()

submission['bidprice'] = bids

submission.to_csv('../submissions/prud_test.csv', index=False)

r = []

import requests
with open('../submissions/prud_test.csv', 'rb') as f:
    r += [requests.post('http://deepmining.cs.ucl.ac.uk/api/upload/wining_criteria_2/31pr3HIVQEC9',
                      files={'file': f})]

print(r[-1].content.decode('utf-8'))