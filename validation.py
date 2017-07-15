from itertools import chain

import numpy as np
import pandas as pd

from data_provider import convert_to_test


def smape(truth, predictions):
    # type: (np.ndarray, np.ndarray) -> float
    """Symmetric mean absolute percentage error."""
    assert truth.shape == predictions.shape, \
        'Ground truth and predictions must have the same shape'
    assert truth.ndim == 1, 'SMAPE expects 1d arrays on input'

    # In order to ignore the true divide errors and remove nans, set errstate
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.abs(truth - predictions) / ((np.abs(truth) + np.abs(predictions)) / 2)
        r[~np.isfinite(r)] = 0
    r = np.mean(r) * 100

    return r


def validate_on_last_days(n_days):
    pass


def forward_chaining(data, folds):
    # type: (pd.DataFrame, int) -> Generator[pd.DataFrame]
    assert data.columns.values[0] == 'Page'
    splits = np.array_split(data.columns.values[1:], folds + 1)

    for i in range(1, folds):
        train_indices = list(chain(*splits[:i]))
        test_indices = splits[i + 1]
        yield data[train_indices], data[test_indices]


def validate_forward_chaining():
    pass


if __name__ == '__main__':
    train = pd.read_csv('data/train_1.csv')
    for train, test in forward_chaining(train, 5):
        print(train.shape, test.shape)
