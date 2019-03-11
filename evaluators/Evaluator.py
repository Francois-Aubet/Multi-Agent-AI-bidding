
# for potential future theading uses:
# import threading
# import time

def _evaluate(df_y, alg_bids):
    number_of_clicks = 0
    click_through_rate = 0
    total_money_paid = 0
    avr_CPM = 0
    avr_CPC = 0
    number_won_bids = 0

    # we define the budget of the algorithm
    budget = (6250000. / 303375.) * len(df_y)

    payprices = df_y['payprice'].values
    clicks = df_y['click'].values
    # we iterate through the test set.
    #   for each impression we get the algorithm bid
    # we then look at if it wins the bid and if so we reduce its budget and augment the metrics
    for i in range(len(df_y)):

        # we get the bid:
        alg_bid = alg_bids[i]

        # in case the algorithm is not able to bet that much:
        if alg_bid > budget:
            # print("stoped after " + str(i) + " bids")
            break
            # continue
        # TODO: confirm what happens at the first too high bid

        # we check if the bid wins the impression:
        other_bid = payprices[i]
        if alg_bid > other_bid:
            # the cost is substracted from the budget
            budget -= other_bid

            # we update the metrics:
            number_of_clicks += clicks[i]
            total_money_paid += other_bid

            number_won_bids += 1

    total_money_paid = float(total_money_paid) / 1000.

    if number_of_clicks > 0:
        avr_CPC = total_money_paid / float(number_of_clicks)
        click_through_rate = float(number_of_clicks) / number_won_bids
        avr_CPM = float(total_money_paid) / (float(number_won_bids) / 1000.)
    else:
        avr_CPC = 0
        click_through_rate = 0
        avr_CPM = 0

    # print(number_of_clicks)
    # print(click_through_rate)
    # print(budget)
    # print(number_won_bids)

    return number_of_clicks, click_through_rate, total_money_paid, avr_CPM, avr_CPC, number_won_bids


class Evaluator():
    """ 
    The main class to do a standard evauation of the non competitive bidding.
    """

    def __init__(self, algoInstance, data_handler):
        """ 
        Initialize the evaluator for one algorithm class.

        :param algoClass:
        :param data_handler:
        """

        # save the input arguments:
        self._algoClass = algoInstance.__class__
        self._data_hanlder = data_handler

        # save an instance of the algorithm
        self._algo = algoInstance

        # we train it using the data in the datahandler
        # self._algo.train(data_handler)

    def evaluate(self):
        """
        Evaluate the algorithm on a test set and computes the different metrics.

        @: return: number of clicks
        @: return: Click-Through Rate
        @: return: Total Money Paid
        @: return: Average CPM (Cost Per Mille)
        @: return: Average CPC (Cost Per Click)
        """
        _, _, test_x, test_y, _ = self._data_hanlder.get_datasets()

        # train the algorithm
        self._algo.train(self._data_hanlder)
        alg_bids = self._algo.predict(test_x, mode='valid')

        number_of_clicks, click_through_rate, total_money_paid, avr_CPM, avr_CPC, number_won_bids = _evaluate(test_y, alg_bids)

        return number_of_clicks, click_through_rate, total_money_paid, avr_CPM, avr_CPC, number_won_bids
