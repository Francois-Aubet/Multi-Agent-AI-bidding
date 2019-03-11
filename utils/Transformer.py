import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

"""
under development, api might change any time
"""


class CategoryCutter:
    def __init__(self, feature_names, no_of_classes=None, min_freq=-1, verbose=True):
        self.no_of_classes = no_of_classes
        self.min_freq = min_freq
        self.feature_names = feature_names
        self.allowed_dict = {}
        self.verbose = verbose

    @staticmethod
    def _fixcolumn(x):
        return x.astype(str).str.replace("nan", "NONE").fillna("NONE")

    def fit_one(self, fname, data):
        counts = self._fixcolumn(data[fname]).value_counts()
        counts1 = counts[counts >= self.min_freq]
        counts2 = counts1.iloc[:self.no_of_classes]
        if self.verbose:
            print("####### categories: %s - orig: %d, after min freq : %d" %
                  (fname, len(counts), len(counts1))
                  )
        self.allowed_dict[fname] = counts2.index

    def transform_one_inplace(self, fname, data):
        data[fname] = self._fixcolumn(data[fname])
        data.loc[~data[fname].isin(self.allowed_dict[fname]), fname] = "OTHER"

    def fit(self, data):
        for fname in self.feature_names:
            self.fit_one(fname, data)

    def transform(self, data):
        data = data.copy()
        for fname in self.feature_names:
            self.transform_one_inplace(fname, data)

        return data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


class OneHotPandasEncoder:
    def __init__(self, feature_names, verbose=True, prefix=""):
        self.feature_names = feature_names
        self.verbose = verbose
        self.dummy_names = None
        self.prefix = prefix
        self._num_columns = None

    def _make_dummies(self, data) -> pd.DataFrame:
        data = data.copy()
        num_columns = data._get_numeric_data().columns.intersection(self.feature_names)
        data[num_columns] = data[num_columns].astype("float32")
        data[self.feature_names] = data[self.feature_names].astype(str)
        data1 = pd.get_dummies(data, columns=self.feature_names, prefix_sep=self.prefix)
        return data1

    def fit_transform(self, data):
        data1 = self._make_dummies(data)
        self.dummy_names = data1.columns.difference(data.columns)
        return data1

    def transform(self, data):
        data1 = self._make_dummies(data)

        col_to_add = self.dummy_names.difference(data1.columns)
        for c in col_to_add:
            data1[c] = 0

        select_cols = data.columns.difference(self.feature_names).union(self.dummy_names)

        if self.verbose:
            print("ohe: not found categories %s" % col_to_add.tolist())
            print("ohe: removed extra categories %s" %
                  data1.columns.difference(select_cols).tolist())

        return data1[select_cols]


class BagOfTags:
    # for usertag preprocessing
    vect = None

    def fit(self, series):
        tags_str = series.fillna('').str.split(',').str.join(' ')
        vect = CountVectorizer()
        vect.fit(tags_str)
        self.vect = vect

    def transform(self, series):
        if self.vect is None:
            raise RuntimeError('Please fit before applying transformation. ')
        return self.vect.transform(series.fillna('').str.split(',').str.join(' '))

    def fit_transform(self, series):
        self.fit(series)
        return self.transform(series)


class AddColumn:
    def __init__(self, fn, new_col_name):
        self.fn = fn
        self.new_col_name = new_col_name

    def transform(self, data):
        data = data.copy()
        data[self.new_col_name] = self.fn(data)
        return data

    def fit_transform(self, data):
        return self.transform(data)


class MeanLabelEncoder:
    def __init__(self, feature_names, label, global_mean, damper, prefix="_ctr"):
        self.feature_names = feature_names
        self.label = label
        self.label_mean_dict = {}
        self.global_mean = global_mean
        self.damper = damper
        self.prefix = prefix

    def fit_one(self, fname, data):
        mean_dict = data.groupby(fname)[self.label].agg(
            lambda x: (np.sum(x) / self.damper + self.global_mean) / (len(x) / self.damper + 1)
        ).to_dict()

        self.label_mean_dict[fname] = mean_dict

    def transform_one_inplace(self, fname, data):
        mean_dict = self.label_mean_dict[fname]
        data[fname + self.prefix] = data[fname].apply(lambda x: mean_dict.get(x, self.global_mean))

    def fit(self, data):
        for fname in self.feature_names:
            self.fit_one(fname, data)

    def transform(self, data):
        data = data.copy()
        for fname in self.feature_names:
            self.transform_one_inplace(fname, data)

        return data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


def transform_all(data, transformers, fit=False):
    d = data

    for f in transformers:
        if fit:
            d = f.fit_transform(d)
        else:
            d = f.transform(d)
    return d
