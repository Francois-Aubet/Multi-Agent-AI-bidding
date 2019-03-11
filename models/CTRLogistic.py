import numpy as np
from scipy.sparse import bmat
from sklearn.preprocessing import OneHotEncoder
from sklearn import linear_model

from models.IModel import IModel
from utils.Transformer import CategoryCutter, BagOfTags
from sklearn.preprocessing import StandardScaler


class CTRLogistic(IModel):

    features = ['slotformat',
                 'adexchange',
                 'os',
                 'weekday',
                 'advertiser',
                 'browser',
                 'slotvisibility',
                 'slotheight',
                 'keypage',
                 'slotwidth',
                 'hour',
                 'region',
                 'useragent',
                 'creative',
                 'slotprice',  # low ranking feature imp but more clicks in the end
                 # 'slotprice_z',
                 'city',
                 'domain',
                 'slotid',
                 'IP',  # 'IP split',
                 # 'usertag',
                 'bag of tags',
                 'url']

    num_features = ['slotprice_z']

    # preprocesssing vars
    bot = cc = ohe = ss = None

    class_ratio = None
    model = None

    def __init__(self, features=None, C=0.001, max_iter=100, min_cat_freq=None, calibrate_prob=True, verbose=True):
        super(CTRLogistic, self).__init__()

        if features is not None:
            self.features = features
        self.C = C
        self.max_iter = max_iter
        self.calibrate_prob = calibrate_prob
        self.verbose = verbose
        self.min_cat_freq = min_cat_freq
        self.model = linear_model.LogisticRegression(C=C, class_weight='balanced', solver='lbfgs', max_iter=max_iter)

    def engineer_features(self, df):
        features = self.features
        # add simple features
        if 'useragent' in df.columns:
            if 'os' in features:
                df['os'] = df.useragent.str.split('_').str[0]
            if 'browser' in features:
                df['browser'] = df.useragent.str.split('_').str[1]
            if 'useragent' not in features:
                df.drop(columns='useragent', inplace=True)

        if 'IP' in df.columns and 'IP split' in features:
            ip_subs = df.IP.str.split('.', expand=True)
            if 'IP' not in features:
                df.drop(columns='IP', inplace=True)
            return df.merge(ip_subs, left_index=True, right_index=True)
        else:
            return df


    def fit(self, df_in):
        # returns sparse matrix
        features = self.features.copy()
        f2 = [i for i in df_in.columns if i in features]
        f2 += [i for i in ['useragent', 'IP', 'slotprice'] if i in df_in.columns]  # add temporarily
        # f2 += [i for i in ['useragent', 'IP'] if i in df_in.columns]  # add temporarily
        f2 = list(set(f2))

        df = self.engineer_features(df_in[f2].copy())
        ss = self.ss
        if 'slotprice_z' in features and 'slotprice' in df.columns:
            ss = StandardScaler()
            df['slotprice_z'] = ss.fit_transform(np.log1p(df.slotprice.to_frame()))

        # region categorical
        # categorical data + usertag + one-hot encoding
        if self.min_cat_freq is None:
            cc = CategoryCutter([i for i in features if i in df.columns and i not in self.num_features],
                            verbose=self.verbose)
        else:
            cc = CategoryCutter([i for i in features if i in df.columns and i not in self.num_features],
                                min_freq=self.min_cat_freq, verbose=self.verbose)

        df_cat = cc.fit_transform(df)
        ohe = OneHotEncoder(handle_unknown='ignore')
        df_cat = ohe.fit_transform(df_cat)

        # merge sparse matrices
        if 'usertag' in df_in.columns and 'bag of tags' in features:
            bot = BagOfTags()
            usertag = bot.fit_transform(df_in.usertag)
            df_out = bmat([[df_cat, usertag]])
        else:
            bot = None
            df_out = df_cat
        # endregion

        self.bot = bot
        self.cc = cc
        self.ohe = ohe
        self.ss = ss

        return df_out

    def transform(self, df_in):
        bot = self.bot
        cc = self.cc
        ohe = self.ohe
        features = self.features

        if cc is None or ohe is None:
            raise RuntimeError('Run fit first!')

        f2 = [i for i in df_in.columns if i in features]
        f2 += [i for i in ['useragent', 'IP', 'slotprice'] if i in df_in.columns]  # add temporarily
        # f2 += [i for i in ['useragent', 'IP'] if i in df_in.columns]  # add temporarily
        f2 = list(set(f2))

        df = self.engineer_features(df_in[f2].copy())
        if self.ss is not None and 'slotprice' in df.columns:
            df['slotprice_z'] = self.ss.transform(np.log1p(df.slotprice.to_frame()))

        df_cat = cc.transform(df)
        df_cat = ohe.transform(df_cat)

        if bot is not None:
            usertag = bot.transform(df_in.usertag)
            df_out = bmat([[df_cat, usertag]])
        else:
            df_out = df_cat

        return df_out

    def calibrate_probability(self, p):
        return p / (p + (1 - p) * self.class_ratio)

    def train(self, data_handler, submission=False):
        train_x, train_y, _, _, _ = data_handler.get_datasets()

        X = self.fit(train_x)
        self.model.fit(X, train_y.click)

        if self.calibrate_prob:
            self.class_ratio = (train_y.click == 0).sum() / (train_y.click == 1).sum()

    def predict(self, test_x):
        # returns probabilities for positive class
        X = self.transform(test_x)
        p = self.model.predict_proba(X)[:, 1]
        if self.calibrate_prob:
            return self.calibrate_probability(p)
        else:
            return p
