from tqdm import tqdm
import numpy as np
import joblib

from Metrics import Metrics
from utils import sample_match_valid


class MultiEvaluatorOnline():
    """ 
    The evaluation of the multi agent competitive bidding. Using winning criteria 2.

    """

    def __init__(self, algo_list, data_handler, use_pretrained=True):
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

    def evaluate(self, use_train=False, numb_epochs=1, verbose=True):
        """
        Evaluate the algorithm on a test set and computes the different metrics.

        :param use_train: specify if we use samples from the test set to run evaluations

        @: return: number of clicks
        @: return: Click-Through Rate
        @: return: Total Money Paid
        @: return: Average CPM (Cost Per Mille)
        @: return: Average CPC (Cost Per Click)
        """
        submission_mode = True
        csv_bid_list = []

        train_x, train_y, test_x, test_y, true_test_x = self._data_hanlder.get_datasets()

        winner_i = -1
        train_epoch = 0
        # this loop is placed here in orther to do multiple runs of evaluation
        while winner_i == -1 or use_train:

            if use_train:
                impress_x, impress_y = sample_match_valid(train_x, train_y, test_y)
                impress_x = impress_x.assign(payprice=impress_y.payprice)

                if train_epoch > numb_epochs:
                    break
                train_epoch += 1
            elif submission_mode:
                impress_x, impress_y = true_test_x, test_y
                print(impress_x.head())
            else:
                impress_x, impress_y = test_x, test_y

            # get the bids from each algorithm and create a Metrics object for each of them:
            metric_list = []
            for algo in self._algo_list:
                metric_list.append(Metrics(algo.alg_name, len(impress_y)))

            payprices = impress_y['payprice'].values
            clicks = impress_y['click'].values

            for i in tqdm(range(len(impress_y))):

                impression = impress_x.loc[[i]]
                # we get the bid of each algorithm for that impression:
                bids_impr = np.zeros((len(self._algo_list), 1))
                for j in range(len(self._algo_list)):
                    bids_impr[j] = self._algo_list[j].predict_single(impression, metric_list[j])
                    metric_list[j].reset_bid_var(bids_impr[j])
                    if i % 3000 == 0 and verbose:
                        #print("Iteration {}".format(i))
                        #print("Algorithm {}".format(self._algo_list[j].alg_name))
                        print(metric_list[j].get_short_metrics())

                if submission_mode:
                    csv_bid_list.append([impression["bidid"].values[0], bids_impr.ravel()[0]])

                # a loop designed to find the winner of the bid
                found_winner = False
                while not found_winner:
                    winner_i = np.random.choice(np.flatnonzero(bids_impr == bids_impr.max()))#np.argmax(bids_impr)

                    if bids_impr[winner_i] == 0:
                        # none can affort this bid...
                        break

                    if metric_list[winner_i].budget >= bids_impr[winner_i] and bids_impr[winner_i] > payprices[i] and not submission_mode:
                        found_winner = True
                    elif submission_mode and metric_list[winner_i].budget >= bids_impr[winner_i]:
                        found_winner = True
                    else:
                        bids_impr[winner_i] = 0

                if not found_winner:
                    # check if the budget of all the algorithms ran out:
                    should_stop = False
                    for metric in metric_list:
                        should_stop = should_stop or (metric.budget > 80)
                    if not should_stop:
                        break
                    # pass

                if found_winner:
                    # we get the pay price which is the second lowest bid:
                    bids_impr_tmp = bids_impr.copy()
                    bids_impr_tmp[winner_i] = 0
                    if submission_mode:
                        payprice = np.max(bids_impr_tmp)
                    else:
                        payprice = max(np.max(bids_impr_tmp), payprices[i])

                    # once the winner is found, we update its state:
                    metric_list[winner_i].update_won_bid(payprice, clicks[i])
 

            result_list = []
            for metr in metric_list:
                result_list.append(metr.compute_metrics())

        return result_list, csv_bid_list

        # bids_impr_li = joblib.Parallel(n_jobs=8, verbose=0)(
        #     joblib.delayed(algo.predict_single)(impression) for algo in self._algo_list
        # )
        # bids_impr = np.array(bids_impr_li)

        # _ = joblib.Parallel(n_jobs=8, verbose=0)(
        #     joblib.delayed(metric.reset_bid_var)() for metric in metric_list
        # )

        # print(bids_impr)
        # print(bids_impr_li)
        # print('--')
