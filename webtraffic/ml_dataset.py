import json
from os.path import join

import fire
import numpy as np
import pandas as pd

from data_provider import get_date_columns, TRAIN_DATA, DATA_DIR

ML_DATASET_DIR = join(DATA_DIR, 'ml')
ML_DATASET = join(ML_DATASET_DIR, 'ml_train_1.csv')
ML_DATASET_INFO = join(ML_DATASET_DIR, 'ml_train_1_data.json')
ML_TRAIN, ML_VALIDATION, ML_TEST = [
    join(ML_DATASET_DIR, 'split_%s.csv' % s)
    for s in ('train', 'validation', 'test')
]

LAG_DAYS = 30


def get_lag_columns(lag_days=LAG_DAYS):
    return list(reversed(['lag_%d' % i for i in range(1, lag_days + 1)]))


def prepare(fname=False, n_last_days=40, lag_days=LAG_DAYS):
    """Prepare a dataset split that we can use with classical ml approaches.

    Creates a dataset using each Page/date combination as a datapoint, adds
    lag variables to each, and some other useful timeseries information.

    Since this produces a ridiculous amount of datapoints, and memory is often
    limited, we specify how many last days to use with the `n_last_days`
    parameter.

    Parameter
    ---------
    fname : Optional[int]
        The file where the initial data is located. This data will be used
        to generate the dataset.
    n_last_days : Optional[int]
        How many days (from the last date backwards) to turn into data points.
    lag_days : Optional[int]
        How many lag variables to generate for each datapoint.

    """
    data = pd.read_csv(fname or TRAIN_DATA)

    date_columns = get_date_columns(data)
    used_data = data[['Page'] + date_columns[-n_last_days:]]

    flattened = pd.melt(used_data, id_vars='Page', var_name='date',
                        value_name='Visits')
    # Drop any columns where the target value is unknown
    flattened.dropna(how='any', inplace=True)

    date_indices = {d: i for i, d in enumerate(date_columns)}

    # We will need the original data page indices and to set the index to page
    data['page_indices'] = data.index
    data.set_index('Page', inplace=True)
    flattened.set_index('Page', inplace=True)

    flattened['date_indices'] = flattened['date'].apply(date_indices.get)
    flattened = flattened.join(data['page_indices'])

    for lag in range(1, lag_days + 1):
        flattened['lag_%d' % lag] = data[date_columns].values[
            flattened['page_indices'],
            flattened['date_indices'] - lag
        ]

    # Remove columns used for indexing
    flattened.drop(['page_indices', 'date_indices'], inplace=True, axis=1)

    # Since we're not lacking in training data, drop any row with NaN lag vars
    flattened.dropna(how='any', inplace=True)

    # Add page mean and median
    flattened['ts_median'] = data.median(axis=1)
    flattened['ts_mean'] = data.mean(axis=1)
    flattened['ts_std'] = data.std(axis=1)

    # Set correct dtypes
    flattened['date'] = flattened['date'].astype('datetime64[ns]')
    flattened['Visits'] = flattened['Visits'].astype(np.float64)
    lag_columns = get_lag_columns(lag_days)
    flattened[lag_columns] = flattened[lag_columns].astype(np.float64)

    # Create the appropriate files
    flattened.to_csv(ML_DATASET)
    train, *_ = make_splits(flattened)

    make_info_file(train, lag_days)


def make_splits(data):
    split = np.split(
        data.sample(frac=1), [int(.6 * len(data)), int(.95 * len(data))]
    )
    for dataset, fname in zip(split, (ML_TRAIN, ML_VALIDATION, ML_TEST)):
        dataset.to_csv(fname)

    return split


def make_info_file(data, lag_days=LAG_DAYS):
    # TODO Add things here as needed
    with open(ML_DATASET_INFO, 'w') as f:
        f.write(json.dumps({}))


def get_info_file():
    with open(ML_DATASET_INFO, 'r') as f:
        return json.load(f)


def lag_test_set_fname(lag_days=LAG_DAYS):
    return join(ML_DATASET_DIR, 'lag_test_set_%d.csv' % lag_days)


if __name__ == '__main__':
    fire.Fire()
