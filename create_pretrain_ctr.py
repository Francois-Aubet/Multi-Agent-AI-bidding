import pickle

from CTRModels.CTRLogistic import CTRLogistic
from CTRModels.CTRXGboost import CTRXGboost
from DataHandler import DataHandler
import numpy as np

from models.MPXGBoost import MPXGboost

train_set_path = '../dataset/we_data/train.csv'
vali_set_path = '../dataset/we_data/validation.csv'
test_set_path = '../dataset/we_data/test.csv'

# this is just a thing to make the reding of the
debug_mode = False

# create a data hanlder instance:
data_handler = DataHandler(train_set_path, vali_set_path, test_set_path, debug_mode)

train_x, train_y, valid_x, valid_y, test_x = data_handler.get_datasets()

#ctr_model = CTRXGboost()
ctr_model = CTRLogistic()

ctr_model.train(data_handler)
ctr_train = ctr_model.predict(train_x)
ctr_valid = ctr_model.predict(valid_x)
ctr_test = ctr_model.predict(test_x)

ctr_all = np.concatenate((ctr_train, ctr_valid, ctr_test))
bidid_all = np.concatenate((train_x.bidid.values, valid_x.bidid.values, test_x.bidid.values))

ctr_dict = dict(zip(bidid_all, ctr_all))





file_class = open('prepared_MP_XGboost', 'w+b')
pickle.dump(ctr_dict, file_class)
