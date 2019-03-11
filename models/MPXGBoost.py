from sklearn.preprocessing import OneHotEncoder
from models.IModel import IModel
import numpy as np
import xgboost as xgb
import pandas as pd
from Transformer import AddColumn, transform_all, BagOfTags, CategoryCutter


class MPXGboost(IModel):

    def __init__(self, calibrate_prob=True):

        self.cat_features = [
            'slotformat',
            'adexchange',
            'os',
            'weekday',
            'advertiser',
            'browser',
            'slotvisibility',
            'keypage',
            'hour',
            'region',
            'creative',
            'domain',
            'city',
            'slotid'
        ]
        self.continous_features = [
            'slotprice',
            'slotwidth',
            'slotheight',
            # 'domain_ctr',
            # 'city_ctr',
            # 'slotid_ctr',
        ]

        self.calibrate_prob = calibrate_prob

    def transform(self, df_x, fit):
        pre_df_x = transform_all(df_x, self.pipelines, fit=fit)

        if fit:
            bag_tags = self.bot.fit_transform(pre_df_x.usertag)
            cat_ohe = self.ohe.fit_transform(pre_df_x[self.cat_features])
        else:
            bag_tags = self.bot.transform(pre_df_x.usertag)
            cat_ohe = self.ohe.transform(pre_df_x[self.cat_features])

        X_out = hstack([pre_df_x[self.continous_features].values, cat_ohe, bag_tags])

        return X_out

    def underSampling(self, train_x, p):
        neg_x = train_x[(train_x.click == 0)]
        pos_x = train_x[(train_x.click == 1)]

        ret = pd.concat([neg_x.sample(int(len(neg_x) * p)), pos_x], ignore_index=True)
        ret = ret.sample(len(ret))

        return ret

    # def calibrate_probability(self, p):
    #     return p / (p + (1 - p) / self.down_sampling_rate)

    def train(self, data_handler, submission=False):
        train_x, train_y, vali_x, vali_y, test_x = data_handler.get_datasets()
        train_x = train_x.assign(payprice=train_y['payprice'])
        train_x = train_x[train_x.payprice > 0]
        vali_x = vali_x.assign(payprice=vali_y['payprice'])
        vali_x = vali_x[vali_x.payprice > 0]

        self.pipelines = [AddColumn(lambda df: df.useragent.str.split("_"), "os_browser"),
                          AddColumn(lambda df: df.os_browser.apply(lambda x: x[0]), "os"),
                          AddColumn(lambda df: df.os_browser.apply(lambda x: x[1]), "browser"),
                          CategoryCutter(feature_names=self.cat_features, min_freq=100)]
        # MeanLabelEncoder(['domain', 'slotid', 'city'], 'click', train_x['click'].mean(), 100)]

        self.bot = BagOfTags()
        self.ohe = OneHotEncoder(handle_unknown='ignore')

        train_x_sample = train_x.sample(int(len(train_x)))

        X_train = self.transform(train_x_sample, fit=True)
        y_train = train_x_sample['payprice'].astype(np.float32)

        X_vali = self.transform(vali_x, fit=False)
        y_vali = vali_x['payprice'].astype(np.float32)

        # scale_pos_weight = (1 - y_train).sum() / y_train.sum()
        self.model = xgb.XGBRegressor(n_estimators=50,
                                      objective="reg:gamma",
                                      random_state=42,
                                      eval_metric="gamma-nloglik",
                                      max_depth=20,
                                      colsample_bytree=0.8,
                                      subsample=1.0,
                                      silent=True)

        self.model.fit(X_train, y_train,
                       early_stopping_rounds=10,
                       eval_set=[(X_train, y_train), (X_vali, y_vali)])

    def predict(self, test_x):
        # returns probabilities for positive class
        X = self.transform(test_x, fit=False)

        y_pred = self.model.predict(X)

        return y_pred
