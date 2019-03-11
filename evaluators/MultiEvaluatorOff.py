from tqdm import tqdm
import numpy as np

from evaluators.Metrics import Metrics




class MultiEvaluatorOff():
    """ 
    The evaluation of the multi agent competitive bidding.
    """

    def __init__(self, algo_list, data_handler, use_pretrained = True):
        """ 
        Initialize the evaluator.

        :param algo_list: the list of the aglo class that we want to let compete
        :param data_handler:    the data
        :param use_pretrained:  boolean to say if we should use the pretrained version of the algorithms
        """

        # save the input arguments:
        self._algo_list = algo_list
        self._data_hanlder = data_handler
        self._used_pretrained = use_pretrained

        # depending on use_pretrained we load the  models or train them
        if use_pretrained:
            # load algos...
            pass
        else:
            for algo in self._algo_list:
                # we train it using the data in the datahandler
                algo.train(self._data_hanlder)



    def evaluate(self, use_train = False):
        """
        Evaluate the algorithm on a test set and computes the different metrics.

        :param use_train: specify if we use samples from the test set to run evaluations

        @: return: number of clicks
        @: return: Click-Through Rate
        @: return: Total Money Paid
        @: return: Average CPM (Cost Per Mille)
        @: return: Average CPC (Cost Per Click)
        """
        train_x, train_y, test_x, test_y, _ = self._data_hanlder.get_datasets()

        winner_i = -1

        # this loop is placed here in orther to do multiple runs of evaluation
        while winner_i == -1 or use_train:

            if use_train:
                # TODO: sample from train set in order to have multiple epochs
                pass
            else:
                impress_x, impress_y = test_x, test_y

            # get the bids from each algorithm and create a Metrics object for each of them:
            metric_list = []
            bids_list = []
            for algo in self._algo_list:
                bids = algo.predict(impress_x)
                bids_list.append(bids)
                metric_list.append(Metrics(algo.alg_name,len(impress_y)))


            payprices = impress_y['payprice'].values
            clicks = impress_y['click'].values

            bid_arr = np.array(bids_list)


            for i in tqdm(range(len(impress_y))):
                
                # save state:
                #for metric in metric_list:
                #    metric.reset_bid_var()


                # we get all the bids for that epoch:
                bids_impr = bid_arr[:,i]


                # a loop designed to find the winner of the bid
                found_winner = False
                while not found_winner:
                    winner_i = np.argmax(bids_impr)

                    if bids_impr[winner_i] == 0:
                        # none can affort this bid...
                        break

                    if metric_list[winner_i].budget >= bids_impr[winner_i] and bids_impr[winner_i] > payprices[i]:
                        found_winner = True
                    else:
                        bids_impr[winner_i] = 0

                if not found_winner:
                    # the budget of all the algorithms ran out
                    #break
                    pass
                #print(winner_i)

                if found_winner:
                    # we get the pay price which is the second lowest bid:
                    bids_impr_tmp = bids_impr.copy()
                    bids_impr_tmp[winner_i] = 0
                    payprice = max(np.max(bids_impr_tmp),payprices[i])

                    # once the winner is found, we update its state:
                    metric_list[winner_i].update_won_bid(payprice,clicks[i])

            result_list = []
            for metr in metric_list:
                result_list.append(metr.compute_metrics())


        return result_list







