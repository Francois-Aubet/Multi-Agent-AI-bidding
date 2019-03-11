
from algorithms.CEAlgorithm import CEAlgorithm
from algorithms.ConstantBid import ConstantBid, ConstantBidCE
from algorithms.RandomBid import RandomBid
from Evaluator import Evaluator
from MultiEvalualOnline import MultiEvaluatorOnline
from MultiEvaluatorOff import MultiEvaluatorOff
from DataHandler import DataHandler

import matplotlib.pyplot as plt
import numpy as np

# just change the pandas printing options
import pandas as pd

import dill as pickle

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 1000)

# define the places of the datasets:

train_set_path = '../dataset/we_data/train.csv'
vali_set_path = '../dataset/we_data/validation.csv'
test_set_path = '../dataset/we_data/test.csv'



# this is just a thing to make the reding of the 
debug_mode = True

# create a data hanlder instance:
#data_handler = DataHandler(train_set_path, vali_set_path, test_set_path, debug_mode)



# multi agent thingy:
n = 50

step = 5

max_n = 101

if False:

    bid_algo = RandomBid()

    evalua = Evaluator(bid_algo, data_handler)

    list_res = []

    for i in range(20):
        list_res.append(evalua.evaluate())

    arr_res = np.array(list_res)

    for i in range(len(list_res[0])):
        print(arr_res[:,i].mean())





    res_list1 = []
    res_list2 = []
    res_list3 = []
    res_list4 = []
    res_list5 = []

    while n < max_n:
        #base = 80
        algo_list = []
        for i in range(n):
            algo_list.append(RandomBid())

        #multi_eval = MultiEvaluatorOnline([algo1,algo2],data_handler)
        multi_eval = MultiEvaluatorOff(algo_list,data_handler)

        result = multi_eval.evaluate()

        #print(result)
        tot_numb_clicks = 0
        tot_wins = 0
        tot_spent = 0
        tot_cpm = 0
        tot_cpc = 0
        for dics in result:
            tot_numb_clicks += dics['number_of_clicks']
            tot_wins += dics['number_won_bids']
            tot_spent += dics['total_money_paid']
            #print()

        #print(tot_numb_clicks,tot_wins,tot_spent)
        res_list1.append(tot_numb_clicks)
        res_list2.append(tot_wins)
        res_list3.append(tot_spent)
        res_list4.append(float(tot_spent) / (float(tot_wins) / 1000.))
        res_list5.append(tot_spent / float(tot_numb_clicks))

        n += step




#result = [res_list1,res_list2,res_list3,res_list4,res_list5]
#file_class = open('saving_the_meas4','w+b')
#pickle.dump(result, file_class)


file_class = open('saving_the_meas4','r+b')
list_meas = pickle.load(file_class)

res_list1 = list_meas[0]
res_list2 = list_meas[1]
res_list3 = list_meas[2]
res_list4 = list_meas[3]
res_list5 = list_meas[4]

print(res_list1)
res_list1 = [112, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113]

for i in range(len(res_list1)):
    res_list5[i] = res_list3[i] / float(res_list1[i])


#plt.plot(result[0]["budget_history"])

plt.rcParams.update({'font.size': 16})

fig_size = (8.5,5.3)
plt.figure(figsize=fig_size)
plt.plot(range(50,max_n,step),res_list1)
plt.xlabel("Number agents")
plt.ylabel("Total number of clicks")
plt.savefig("/home/francois/Documents/UCL/Courses/Multi_Agent_AI/cw1/plots/numb_clicks.pdf")

plt.figure(figsize=fig_size)
plt.plot(range(50,max_n,step),res_list2)
plt.xlabel("Number agents")
plt.ylabel("Total number of impressions")
plt.savefig("/home/francois/Documents/UCL/Courses/Multi_Agent_AI/cw1/plots/numb_impressions.pdf")
#plt.show()

plt.figure(figsize=fig_size)
plt.plot(range(50,max_n,step),res_list3)
plt.xlabel("Number agents")
plt.ylabel("Total amount of money spent")
plt.savefig("/home/francois/Documents/UCL/Courses/Multi_Agent_AI/cw1/plots/tot_spend.pdf")
#plt.show()

plt.figure(figsize=fig_size)
plt.plot(range(50,max_n,step),res_list4)
plt.xlabel("Number agents")
plt.ylabel("Average CPM")
plt.savefig("/home/francois/Documents/UCL/Courses/Multi_Agent_AI/cw1/plots/avr_cpm.pdf")
#plt.show()

# plt.figure(figsize=fig_size)
# plt.plot(range(50,max_n,step),res_list5)
# plt.xlabel("Number agents")
# plt.ylabel("Average CPC")
# plt.savefig("/home/francois/Documents/UCL/Courses/Multi_Agent_AI/cw1/plots/numb_clicks.pdf")
#plt.show()
