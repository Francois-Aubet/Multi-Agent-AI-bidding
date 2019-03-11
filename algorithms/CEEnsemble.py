from evaluators.Evaluator import _evaluate
from algorithms.CDFBid import beta_mom, sample_match_valid
import numpy as np
import scipy.stats as stats
import joblib

class CEEnsemble:

    def __init__(self, n_samples=100, p=0.3, max_iter=10, n_jobs=10):
        self.p = p
        self.n_samples = n_samples
        self.max_iter = max_iter
        self.n_jobs = n_jobs

    def train(self, train_y, valid_y, init=None):
        # train_y and valid_y contains bidprices for each algo
        bid_cols = np.array([False if i in ['click', 'bidprice', 'payprice'] else True for i in valid_y.columns])

        n_algos = bid_cols.sum()
        if init is None:
            alpha = np.ones(n_algos)
            beta = alpha.copy()
        else:
            mu = init
            var = stats.beta.var(1, 1)
            alpha = mu * ((mu * (1 - mu)) / var - 1)
            beta = (1 - mu) * ((mu * (1 - mu)) / var - 1)

        wgt = stats.beta.mean(alpha, beta)
        # wgt /= wgt.sum()

        n_samples = self.n_samples
        p = self.p

        for i in range(self.max_iter):
            df_y = sample_match_valid(train_y, valid_y)

            samples = np.array([np.random.beta(a, b, n_samples) for a, b in zip(alpha, beta)])
            # samples /= samples.sum(axis=0, keepdims=1)

            bids = df_y.values[:, bid_cols] @ samples

            results = joblib.Parallel(n_jobs=5, verbose=0)(
                joblib.delayed(_evaluate)(df_y, bids[:, i]) for i in range(bids.shape[1])
            )
            clicks = np.array([r[0] for r in results])

            # get elite samples
            elite = samples[:, clicks.argsort()[::-1][:int(n_samples * p)]]
            # record previous parameters
            prev_wgt = wgt.copy()

            # calculate new mean and std based on the elite samples
            beta_params = [beta_mom(i) for i in elite]
            alpha = np.array([i[0] for i in beta_params])
            beta = np.array([i[1] for i in beta_params])

            wgt = stats.beta.mean(alpha, beta)
            # wgt /= wgt.sum()

            print(clicks[-1], wgt)

            converge_constr = .01
            if np.abs(prev_wgt - wgt).max() < converge_constr:
                print("converge!!")
                break

        print("Optimal weights: " + str(wgt))

        return wgt
