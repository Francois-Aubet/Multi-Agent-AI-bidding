
from DataHandler import DataHandler

#from CTRModels.CTRLogistic import CTRLogistic
#from CTRModels.CTRXGboost import CTRXGboost

#import pickle
import dill as pickle

import matplotlib.pyplot as plt
import numpy as np

# just change the pandas printing options
import pandas as pd




file_class = open('plot_docs/saving_the_meas','r+b')
list_meas = pickle.load(file_class)

list_budget_factor = list_meas[0]
list_base_bid = list_meas[1]


plt.rcParams.update({'font.size': 13})

fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Impressions')
ax1.set_ylabel('budget factor', color=color)
ax1.plot(range(len(list_budget_factor)), list_budget_factor, color=color)
ax1.set_ylim((0.8,1.2))
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:red'
ax2.set_ylabel('mean base bid2', color=color)  # we already handled the x-label with ax1
ax2.plot(range(len(list_base_bid)), list_base_bid, color=color)
#ax2.plot(range(len(true_payprice)), true_payprice, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

 




file_class = open('res_bigRun3','r+b')
result_list = pickle.load(file_class)


plt.rcParams.update({'font.size': 13})

clicks_list = []
for result in result_list:
    #clicks_list.append(result["click_history"])
    plt.plot(range(0,len(result["budget_history"])*20,20),result["budget_history"], linewidth=2.0)

# click_history
# budget_history
# impression_history
#length = range(len(clicks_list[]))

plt.legend(("BidStream","Linear1","Linear2","Closed-Form1","Closed-Form2","CDF1","CDF2","PRUD","Ensemble","Constant","Random"))
locs, other = plt.yticks()
print(locs, other)
plt.yticks([0,1000000,2000000,3000000,4000000,5000000,6000000],[0,"1k","2k","3k","4k","5k","6k"]) 
#plt.show()




for result in result_list:
    print(result["alg_name"],"&",result["number_of_clicks"],"&",result["click_through_rate"],"&",result["total_money_paid"],"&",result["avr_CPM"],"&",result["avr_CPC"],"&",result["number_won_bids"])
    print("\hline")






























