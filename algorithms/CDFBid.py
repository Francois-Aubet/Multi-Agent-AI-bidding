from algorithms.IAlgorithm import IAlgorithm
from models.CTRLogistic import CTRLogistic
from models.CTRXGboost import CTRXGboost
from models.IModel import IModel
from algorithms.CEAlgorithm import CEAlgorithm
from Evaluator import _evaluate

from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np, copy
import scipy.stats as stats
import joblib
import pandas as pd


def get_mean(mean, std):
    return np.exp(mean + (std ** 2) / 2.)


def beta_mom(obs):
    # Method of Moments for beta dist
    mu = np.mean(obs, axis=0)
    var = np.var(obs, axis=0)
    alpha = mu * ((mu * (1 - mu)) / var - 1)
    beta = (1 - mu) * ((mu * (1 - mu)) / var - 1)
    return alpha, beta


def sample_match_valid(train_y, valid_y):
    neg_valid_size = np.sum(valid_y.click == 0)
    pos_valid_size = np.sum(valid_y.click == 1)

    neg_train = train_y[(train_y.click == 0)]
    pos_train = train_y[(train_y.click == 1)]

    ret = pd.concat([neg_train.sample(neg_valid_size), pos_train.sample(pos_valid_size)], ignore_index=True)
    # reshuffle
    ret = ret.sample(len(ret)).reset_index(drop=True)

    return ret


class CDFBid(IAlgorithm):

    alg_name = "CDF"
    _base_params = None
    use_pretrained = False
    w = []
    models = []

    def __init__(self, ceAlgo: CEAlgorithm, ctrModel: IModel, n_ctr=10, test_size=0.3, n_rounds=3,
                 pretrained={}):
        self.ceAlgo = ceAlgo
        self.ctrModel = ctrModel
        self.n_ctr = n_ctr
        self.n_rounds = n_rounds
        self.test_size = test_size

        if len(pretrained)>0:
            self.use_pretrained = True
            self.ctrModel = pretrained['ctrModel']
            self.models = pretrained['models']
            self.w = pretrained['w']
            self._base_params = pretrained['_base_params']

    def train(self, data_handler, mode='train'):
        # mode='valid' to test hyperparameter overfitting on validation
        # when use_pretrained = True, this just tunes hyperparameters according to mode

        ctrModel = self.ctrModel
        ceAlgo = self.ceAlgo
        n_ctr = self.n_ctr

        train_x, train_y, valid_x, valid_y, _ = data_handler.get_datasets()

        use_pretrained = self.use_pretrained
        if use_pretrained:
            models = self.models
            w = self.w
            if isinstance(ctrModel, CTRLogistic):
                X = ctrModel.transform(train_x)
            elif isinstance(ctrModel, CTRXGboost):
                X = ctrModel.transform(train_x, fit=False)

        else:
            if isinstance(ctrModel, CTRLogistic):
                X = ctrModel.fit(train_x)
            elif isinstance(ctrModel, CTRXGboost):
                X = train_x.copy()
            else:
                raise RuntimeError('Unsupported CTRModel')

            models = []
            w = np.zeros(n_ctr)

        tr_preds = []  # calibrated predictions on bootstrapped test folds

        if mode=='train':
            df_y = train_y.copy()
        elif mode=='valid':
            df_y = valid_y.copy()
            vX = ctrModel.transform(valid_x)

        #region: train multiple bootstrappped ctr estimators
        if isinstance(ctrModel, CTRLogistic):
            # optimisations for logistic
            for i in tqdm(range(self.n_ctr)):
                tr_x, _, tr_y, _ = train_test_split(X, train_y.click, test_size=valid_x.shape[0])

                if not use_pretrained:
                    w[i] = (tr_y == 0).sum() / (tr_y == 1).sum()
                    m = ctrModel.model
                    m.fit(tr_x, tr_y)
                    models.append(copy.copy(m))
                else:
                    m = models[i]

                if mode=='train':
                    p = m.predict_proba(X)[:, 1]
                elif mode=='valid':
                    p = m.predict_proba(vX)[:, 1]

                tr_preds.append(p / (p + (1 - p) * w[i]))
                df_y['pred_'+str(i)] = tr_preds[-1]
        else:
            for i in tqdm(range(self.n_ctr)):
                if not use_pretrained:
                    m = ctrModel
                    m.train(data_handler)
                    models.append(copy.copy(m))
                else:
                    m = models[i]

                if mode == 'train':
                    p = m.predict(train_x)
                elif mode == 'valid':
                    p = m.predict(valid_x)

                tr_preds.append(p)
                df_y['pred_' + str(i)] = tr_preds[-1]

        if not use_pretrained:
            self.models = models
            self.w = w
        #endregion

        #region: fit hyperparams
        alpha, beta = beta_mom(tr_preds)
        avgCTR = np.mean(tr_preds)
        conf = 1 - stats.beta(a=alpha, b=beta).cdf(avgCTR)
        conf = np.nan_to_num(conf)

        if not isinstance(ceAlgo, CECDFAlgorithm):
            conf += conf.mean()


        df_y['alpha'] = alpha
        df_y['beta'] = beta
        df_y['multipliers'] = conf

        base_bids = np.zeros(self.n_rounds)
        if isinstance(ceAlgo, CECDFAlgorithm):
            min_bids = np.zeros_like(base_bids)
            for i in range(self.n_rounds):
                base_bid, min_bid = ceAlgo.train(df_y, valid_y)
                base_bids[i] = base_bid
                min_bids[i] = min_bid
            self._base_params = [np.mean(base_bids), np.mean(min_bids)]
        else:
            for i in range(self.n_rounds):
                base_bid = ceAlgo.train(df_y, valid_y)
                base_bids[i] = base_bid
            self._base_params = np.mean(base_bids)
        #endregion

    def predict(self, test_x, mode=None):
        models = self.models

        preds = []
        base_params = self._base_params

        if isinstance(self.ctrModel, CTRLogistic):
            X = self.ctrModel.transform(test_x)
            w = self.w

            for i in range(self.n_ctr):
                p = models[i].predict_proba(X)[:, 1]
                preds.append(p / (p + (1 - p) * w[i]))
        else:
            for i in range(self.n_ctr):
                preds.append(models[i].predict(test_x))
        preds = np.array(preds)

        alpha, beta = beta_mom(preds)
        avgCTR = np.mean(preds)
        conf = 1 - stats.beta(a=alpha, b=beta).cdf(avgCTR)
        conf = np.nan_to_num(conf)

        if len(base_params)<2:
            conf += conf.mean()
            base_bid = base_params
            min_bid = 0
        else:
            base_bid = base_params[0]
            min_bid = base_params[1]

        return conf*base_bid+min_bid


class CECDFAlgorithm:

    def __init__(self, n_samples=100, p=0.3, max_iter=10, n_jobs=10):
        self.p = p
        self.n_samples = n_samples
        self.max_iter = max_iter
        self.n_jobs = n_jobs

    def train(self, train_y, valid_y):
        assert 'multipliers' in train_y.columns, 'train_y must contain multiplier'
        assert 'alpha' in train_y.columns, 'train_y must contain Beta dist parameters'
        assert 'beta' in train_y.columns, 'train_y must contain Beta dist parameters'

        # initialize mean and std
        x = np.log(train_y[train_y.payprice > 0].payprice)
        mean = x.mean()
        std = x.std(ddof=0) * 1.5

        # multipliers is confidence
        conf = train_y.multipliers.values
        mean_b = np.log(conf.mean()) + mean
        std_b = std

        n_samples = self.n_samples
        p = self.p

        pred_cols = train_y.columns.str.contains('pred_')
        for i in range(self.max_iter):
            df_y = sample_match_valid(train_y, valid_y)

            conf = 1 - stats.beta(a=df_y.alpha, b=df_y.beta).cdf(np.mean(df_y.loc[:, pred_cols].values))

            samples = np.random.lognormal(mean, std, n_samples)
            samples_b = np.random.lognormal(mean_b, std_b, n_samples)

            results = joblib.Parallel(n_jobs=5, verbose=0)(
                joblib.delayed(_evaluate)(df_y, base_bid * conf + min_bid) for base_bid, min_bid in zip(samples, samples_b)
            )
            clicks = np.array([r[0] for r in results])
            # get elite samples
            elite = samples[clicks.argsort()[::-1][:int(n_samples * p)]]
            elite_b = samples_b[clicks.argsort()[::-1][:int(n_samples * p)]]
            # update log x
            log_x = np.log(elite)
            # record previous parameters
            prev_mean = mean
            prev_std = std

            # calculate new mean and std based on the elite samples
            mean = log_x.mean()
            std = log_x.std()

            mean_b = np.log(elite_b).mean()
            std_b = np.log(elite_b).std()

            currmean = get_mean(mean, std)
            print(clicks[-1], currmean, get_mean(mean_b, std_b))

            converge_constr = 1
            if np.abs(get_mean(prev_mean, prev_std) - currmean) < converge_constr:
                print("converge!!")
                break

        print("Optimal base bid mean, std: {}, {}".format(mean, std))
        print("Optimal min bid mean, std: {}, {}".format(mean_b, std_b))

        return get_mean(mean, std), get_mean(mean_b, std_b)

