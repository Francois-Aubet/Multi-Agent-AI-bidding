import pandas as pd
import numpy as np


class DataHandler():
    """ 
    The class that handles the preprocessing and such.

    Note: It might be a design mistake to have this as a class since we will probably never ever need 
        more than two instances of it. But it is somewhat convinient this way.
    """

    def __init__(self, train_set_path, vali_set_path, test_set_path, debug_mode):
        """ 
        Initialize the dataset places.
        
        """

        self._train_set_path = train_set_path
        self._vali_set_path = vali_set_path
        self._test_set_path = test_set_path
        
        if debug_mode:
            self.vali_df = pd.read_csv(vali_set_path)
            self.train_df = self.vali_df
            self.test_df = self.vali_df#pd.read_csv(test_set_path)
            #print(len(self.test_df))
            #print(len(self.vali_df))
            #print(self.test_df.loc[[2]])
        else:
            self.train_df = pd.read_csv(train_set_path)
            self.vali_df = pd.read_csv(vali_set_path)
            self.test_df = pd.read_csv(test_set_path)
            
        self.y_cols = ['click', 'bidprice', 'payprice']


    def get_datasets_cross_validate(self, k_fold):
        """
        Returns the preprocessed k-fold train, test splits

        :return: train_test_splits: list[(train_k, test_k)]
        """

        train_n_obs = self.train_df.shape[0]

        # random split corss validation set
        index = np.arange(train_n_obs)
        np.random.shuffle(index)
        fold_indices = np.array_split(index, k_fold)

        train_test_splits = []
        for i in range(k_fold):
            mask = np.ones(train_n_obs, dtype=bool)
            mask[fold_indices[i]] = False
            fold_train = self.train_df.iloc[mask, :]
            fold_test = self.train_df.iloc[fold_indices[i], :]
            train_test_splits.append(
                (fold_train[fold_train.columns.difference(self.y_cols)], fold_train[self.y_cols]
                 , fold_test[fold_test.columns.difference(self.y_cols)], fold_test[self.y_cols])
            )

        return train_test_splits

    def get_datasets(self):
        """
        Returns the preprocessed train, validation and test sets that are saved on the paths.

        :return: train dataframe
        :return: test dataframe
        :return: validation dataframe
        """

        return self.train_df[self.train_df.columns.difference(self.y_cols)], self.train_df[self.y_cols], \
               self.vali_df[self.vali_df.columns.difference(self.y_cols)], self.vali_df[self.y_cols], \
               self.test_df
