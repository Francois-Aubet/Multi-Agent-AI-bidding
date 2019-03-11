from algorithms.IAlgorithm import IAlgorithm
import numpy as np
import math

from evaluators.Metrics import Metrics


class BidStream(IAlgorithm):
    """ The simplest constant bidding algorithm.

    TBD: this is just a test version designed to test the evaluation pipeline.
    """

    alg_name = "BidStream"


    def __init__(self, ctr_pred_table = None, type = 1, ctr_model = None, base_bid = 145):
        """ Constructor of the abstract class.
            Should be called when initializing the child classes.
        """

        # the bissing variables:
        self._base_bid = base_bid

        self._last_bid = 1
        self._last_pCTR = 0.000001

        self._ctr_pred_table = ctr_pred_table
        self._ctr_model = ctr_model

        self._avr_ctr = np.array(list(ctr_pred_table.values())).mean()

        self._type = type

        # parameters of the estimation of other agents:
        # the hyper parameter of the update
        self._moving_recall = 0.9993
        self._moving_rec_second = 0.99993
        self._effitiency_mult = 1000000

        # the prior on the average and std of the market price
        self._mean_opp_bid = 145.0
        self._var_opp_bid = 10
        self._mov_opp_sqr_bid = 16900.0

        self.predicted_opp_bid = 1

        # mean of the last utilities
        self._mean_utility = 5
        self._mean_sqr_utility = 1
        self._var_utility = 1


        self._counter_bids = 0

        # list to log stuff and then plot
        self._list_budget_factor = []
        self._list_base_bid = []
        self._list_eff_factor = []



    def train(self, data_handler):
        """ Training the algorithm using self._train* .
        This needs to be overwritten in every child classes.
        """
        pass


    def predict(self, test_x):
        """ Predicting for a dataset. This needs to be overwritten in every child classes.
        @:return: the bid
        """

        # Note this algorithm is not meant be used in a non sequencial manner, so this is not implemented
        return np.repeat(self._base_bid, test_x.shape[0])


    def initialize_metrics(self,metric: Metrics):
        """ Initialize some stuff on the first prediction. """

        self._avr_budg_impr = metric.budget / metric.n_impression_left

        self._last_factor_buget = 1
        self._last_factor_utility = 1
        self._last_effi = 1




    def predict_single(self,impression,metric: Metrics):
        """ Predicting for a single impression. T
        """
        if self._counter_bids == 0:
            self.initialize_metrics(metric)


        self.update_estimate(metric)

        p_ctr = self._ctr_pred_table[impression.bidid.values[0]]

        self.predicted_opp_bid = (p_ctr / self._avr_ctr) * self._mean_opp_bid

        # this utility
        utility = (p_ctr / self.predicted_opp_bid) * self._effitiency_mult
        factor_utility = (utility / self._mean_utility) ** (1/4)
        if self._counter_bids < 4000 or math.isnan(factor_utility):
            factor_utility = 1
        elif factor_utility < 0.95:
            factor_utility = 0.95
        elif factor_utility > 1.05:
            factor_utility = 1.05

        factor_buget = ( (metric.budget / metric.n_impression_left) / self._avr_budg_impr)

        # the function for the bid price is dependent on the type of the algorithm
        if self._type == 1:
            self._last_bid = self._mean_opp_bid * (p_ctr / self._avr_ctr) * (factor_buget ** 3) #* factor_utility self._base_bid

        elif self._type == 1.1:
            self._last_bid = self._mean_opp_bid * (p_ctr / self._avr_ctr) * np.sqrt(factor_buget)

        else: #if self._type == 2:
            #print(factor_utility,self._mean_utility,(utility),self._last_effi)
            self._last_bid = self._mean_opp_bid * (p_ctr / self._avr_ctr) * factor_buget * (factor_utility)# ** (1/8)) 


        self._list_budget_factor.append(factor_buget)
        self._list_base_bid.append(self._mean_opp_bid)
        self._list_eff_factor.append(factor_utility)

        self._last_factor_buget = factor_buget
        self._last_factor_utility = factor_utility
        self._last_pCTR = p_ctr
        return self._last_bid


    def get_meas_lists(self):
        """ returns the two lists.  """
        return self._list_budget_factor, self._list_base_bid,self._list_eff_factor


    def update_estimate(self,metric: Metrics):
        """ Updates the estimate of the bids of the other agents. """

        # we have two type of update: in case we won the bid and in case we lost it
        if metric.won_last_bid:
            # here we have direct information about the bid of the other agents
            opponent_bid = metric.paid_on_last_bid
            opponent_bid_scaled = opponent_bid / (self._last_pCTR / self._avr_ctr)
            #opponent_bid = opponent_bid_scaled

            # we update the moving averages
            self._mean_opp_bid = self._mean_opp_bid * self._moving_recall + (1 - self._moving_recall) * opponent_bid_scaled
            self._mov_opp_sqr_bid = self._mov_opp_sqr_bid * self._moving_recall + (1 - self._moving_recall) * (opponent_bid_scaled ** 2)

            efficienty = (self._last_pCTR / (opponent_bid+0.001)) * self._effitiency_mult
            self._mean_utility = self._mean_utility * self._moving_recall + (1 - self._moving_recall) * efficienty
            #self._mean_utility = self._mean_utility * self._moving_recall + (1 - self._moving_recall) * np.exp(self._last_pCTR / (opponent_bid+0.001) )
            #self._mean_sqr_utility = self._mean_sqr_utility * self._moving_recall + (1 - self._moving_recall) * ((self._last_pCTR / (opponent_bid+0.01)) ** 2)
            self._last_effi = (self._last_pCTR / (opponent_bid+0.001) ) * self._effitiency_mult

        else:
            # in this case we only know that the oppenent bid more than us
            
            if self.predicted_opp_bid > self._last_bid:
                # we knew that we would loose the auction so this is fine
                pass
            elif self._last_bid != 0.0:
                # we thought that we would win, we want to update our estimation of the bid price

                # first try assuming that the pay price was at least our bid:
                our_bid = self._last_bid / (self._last_pCTR / self._avr_ctr) / (self._last_factor_buget ** 3) # + np.random.rand() *

                self._mean_opp_bid = self._mean_opp_bid * self._moving_rec_second + (1 - self._moving_rec_second) * our_bid
                self._mov_opp_sqr_bid = self._mov_opp_sqr_bid * self._moving_rec_second + (1 - self._moving_rec_second) * (our_bid ** 2)

                efficienty = (self._last_pCTR / (self._last_bid+0.001)) * self._effitiency_mult
                self._mean_utility = self._mean_utility * self._moving_rec_second + (1 - self._moving_rec_second) * efficienty
                #self._mean_utility = self._mean_utility * self._moving_recall + (1 - self._moving_recall) * np.exp(self._last_pCTR / self._last_bid)
                #self._mean_sqr_utility = self._mean_sqr_utility * self._moving_recall + (1 - self._moving_recall) * ((self._last_pCTR / self._last_bid) ** 2)
                self._last_effi = (self._last_pCTR / self._last_bid ) * self._effitiency_mult

        self._var_opp_bid = self._mov_opp_sqr_bid - (self._mean_opp_bid ** 2)
        self._var_utility = self._mean_utility - (self._mean_utility ** 2)

        self._counter_bids += 1
        if self._counter_bids % 3000 == 2:
            print(self._mean_opp_bid,np.sqrt(self._var_opp_bid))
            print(self._mean_utility,np.sqrt(self._var_utility),self._last_factor_utility)
    

            
            

