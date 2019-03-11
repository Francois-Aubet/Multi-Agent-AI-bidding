import joblib
import numpy as np
from Evaluator import _evaluate
import pandas as pd


class CEAlgorithm(object):

    def __init__(self, n_samples, p, max_iter, n_jobs):
        self.p = p
        self.n_samples = n_samples
        self.max_iter = max_iter
        self.n_jobs = n_jobs

    def get_mean(self, mean, std):
        return np.exp(mean + (std ** 2) / 2.)

    def sample_match_valid(self, train_y, valid_y):
        neg_valid_size = np.sum(valid_y.click == 0)
        pos_valid_size = np.sum(valid_y.click == 1)

        neg_train = train_y[(train_y.click == 0)]
        pos_train = train_y[(train_y.click == 1)]

        ret = pd.concat([neg_train.sample(neg_valid_size), pos_train.sample(pos_valid_size)], ignore_index=True)
        # reshuffle
        ret = ret.sample(len(ret)).reset_index()

        return ret

    def train(self, train_y, valid_y, is_random_case=False, is_prud_case=False):
        # assert 'multipliers' in train_y.columns, 'train_y must contain multiplier'

        log_x = np.log(train_y[train_y.payprice > 0].payprice)
        # initialize mean and std

        mean = log_x.mean()
        std = np.sqrt(((log_x - mean) ** 2).mean()) * 1.5  # encourage exploration

        if is_random_case:
            mean = std
            std = std

        if is_prud_case:
            mean = -12.6
            std = 1.5

        for i in range(self.max_iter):

            df_y = self.sample_match_valid(train_y, valid_y)
            if is_prud_case:
                rhos = df_y.rho
                pMP = df_y.pMP
            else:
                multipliers = df_y.multipliers.values

            samples = np.random.lognormal(mean, std, self.n_samples)

            if is_prud_case:

                results = joblib.Parallel(n_jobs=self.n_jobs, verbose=0)(
                    joblib.delayed(_evaluate)(df_y, ((pMP + 300) * rhos.apply(lambda x: x > base_bid)).values) for
                    base_bid in samples
                )
            elif is_random_case:
                results = joblib.Parallel(n_jobs=self.n_jobs, verbose=0)(
                    joblib.delayed(_evaluate)(df_y, multipliers + np.random.uniform(-base_bid, base_bid,
                                                                                    multipliers.shape[0]))
                    for base_bid in samples
                )
            else:
                results = joblib.Parallel(n_jobs=self.n_jobs, verbose=0)(
                    joblib.delayed(_evaluate)(df_y, base_bid * multipliers) for base_bid in samples
                )

            clicks = np.array([r[0] for r in results])
            # get elite samples
            elite = samples[clicks.argsort()[::-1][:int(self.n_samples * self.p)]]
            # update log x
            log_x = np.log(elite)
            # record previous parameters
            prev_mean = mean
            prev_std = std

            # calculate new mean and std based on the elite samples
            mean = log_x.mean()
            std = np.sqrt(((log_x - mean) ** 2).mean())

            currmean = self.get_mean(mean, std)
            print(currmean)
            converge_constr = 1
            if is_prud_case:
                print("prev_mean {}, cur_mean {}".format(prev_mean, mean))
                if np.abs(prev_mean - mean) < .005:
                    print("converge!!")
                    break
            else:
                if is_random_case:
                    converge_constr = 0.1
                if np.abs(self.get_mean(prev_mean, prev_std) - currmean) < converge_constr:
                    print("converge!!")
                    break

        print("Optimal mean, std: {}, {}".format(mean, std))
        print("Optimal train click: {}".format(clicks.mean()))

        return self.get_mean(mean, std)
