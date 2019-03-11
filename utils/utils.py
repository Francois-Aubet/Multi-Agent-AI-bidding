import numpy as np


def sample_match_valid(train_x, train_y, valid_y):
    neg_valid_size = np.sum(valid_y.click == 0)
    pos_valid_size = np.sum(valid_y.click == 1)

    neg_train = train_y[train_y.click == 0]
    pos_train = train_y[train_y.click == 1]

    neg_sample_idx = np.random.choice(neg_train.index, neg_valid_size, replace=False)
    pos_sample_idx = np.random.choice(pos_train.index, pos_valid_size, replace=False)

    merge_idx = np.hstack([neg_sample_idx, pos_sample_idx])
    np.random.shuffle(merge_idx)

    new_train_x = train_x.iloc[merge_idx].reset_index()
    new_train_y = train_y.iloc[merge_idx].reset_index()

    return new_train_x, new_train_y
