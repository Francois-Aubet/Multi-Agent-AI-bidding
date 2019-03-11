class Metrics():
    """ 
    This class is used to record all the metrics of an algorithm as it is evaluated on one competitiv bid run.
    """

    def __init__(self, alg_name, length):

        self._alg_name = alg_name

        # we define the different metric counter
        self.number_of_clicks = 0
        self.total_money_paid = 0
        self.number_won_bids = 0

        # we define the budget of the algorithm
        self.budget = (6250000. / 303375.) * length

        # evolution of the budget over time:
        self.budgets_list = []
        self.click_list = []
        self.won_bid_list = []
        self.n_impression_done = 0
        self.n_impression_left = length

        # create the variable indicating for each run
        self.won_last_bid = False
        self.paid_on_last_bid = 0
        self.click_on_last_bid = 0

        self.average_bid = 0

    def reset_bid_var(self, last_bid):
        """ Resetting the variables that give information about the last impression bid.
        """
        self.won_last_bid = False
        self.paid_on_last_bid = 0
        self.click_on_last_bid = 0

        self.average_bid = (self.average_bid * self.n_impression_done + last_bid) / (self.n_impression_done + 1)


        self.n_impression_done += 1
        self.n_impression_left -= 1
        if self.n_impression_done % 20 == 0:
            self.budgets_list.append(self.budget)
            self.click_list.append(self.number_of_clicks)
            self.won_bid_list.append(self.number_won_bids)


    def update_won_bid(self, payprice, click):
        """
        Performs the update of the metrics in case it won a bid.
        """

        self.budget -= payprice
        self.number_of_clicks += click
        self.total_money_paid += payprice
        self.number_won_bids += 1

        self.won_last_bid = True
        self.paid_on_last_bid = payprice
        self.click_on_last_bid = click

    def compute_metrics(self):
        """ Once the run is done computes the metrics for the alg.
        """

        total_money_paid = float(self.total_money_paid) / 1000.

        if self.number_of_clicks > 0:
            self.click_through_rate = float(self.number_of_clicks) / self.number_won_bids
            self.avr_CPC = total_money_paid / float(self.number_of_clicks)  # float(self.number_of_clicks) / self.total_money_paid
            self.avr_CPM = float(total_money_paid) / (float(self.number_won_bids) / 1000.)  # self.number_won_bids
        else:
            self.click_through_rate = 0
            self.avr_CPC = 0
            self.avr_CPM = 0

        dictionary = {}
        dictionary["alg_name"] = self._alg_name
        dictionary["number_of_clicks"] = self.number_of_clicks
        dictionary["click_through_rate"] = self.click_through_rate
        dictionary["total_money_paid"] = total_money_paid
        dictionary["avr_CPM"] = self.avr_CPM
        dictionary["avr_CPC"] = self.avr_CPC
        dictionary["budget"] = self.budget
        dictionary["number_won_bids"] = self.number_won_bids
        dictionary["average_bid"] = self.average_bid
        dictionary["budget_history"] = self.budgets_list
        dictionary["click_history"] = self.click_list
        dictionary["impression_history"] = self.won_bid_list

        return dictionary


    def get_short_metrics(self):
        """ Retruns small status update. """

        list_metrics = []

        list_metrics.append(self._alg_name)
        list_metrics.append("number_of_clicks")
        list_metrics.append(self.number_of_clicks)
        list_metrics.append("budget")
        list_metrics.append(self.budget)
        list_metrics.append("number_won_bids")
        list_metrics.append(self.number_won_bids)

        return list_metrics