from datetime import date
from itertools import chain

import numpy as np
import pandas as pd

from data_provider import convert_to_test, prepare_test_data, get_date_columns

import logging

logger = logging.getLogger('VALIDATION')


def smape(y, y_hat):
    # type: (np.ndarray, np.ndarray) -> float
    """Symmetric mean absolute percentage error."""
    assert y.shape == y_hat.shape, '`y` and `y_hat` must have the same shape'
    assert y.ndim == 1, 'SMAPE expects 1d arrays on input'

    denominator = np.abs(y) + np.abs(y_hat)
    result = np.abs(y - y_hat) / denominator
    result[denominator == 0] = 0.
    result = np.mean(result) * 200

    return result


def validate_last_n_days(data, learner, n_days=60):
    # type: (pd.DataFrame, Learner, Optional[int]) -> float
    """Train and validate the model on the last `n` days of the dataset."""
    # Training data from first column forth, to skip the `Page` column
    train, test = data[data.columns[1:-n_days]], data[data.columns[-n_days:]]
    # Add the `Page` column to both train and test data
    train['Page'] = test['Page'] = data['Page']

    test = convert_to_test(test)
    test = prepare_test_data(test)

    prediction = learner.fit(train).predict(test)
    prediction.fillna(0, inplace=True)
    return smape(prediction['Actual'], prediction['Visits'])


def forward_chaining(data, folds):
    # type: (pd.DataFrame, int) -> Generator[pd.DataFrame]
    assert data.columns.values[0] == 'Page'
    splits = np.array_split(data.columns.values[1:], folds + 1)

    for i in range(1, folds):
        train_indices = list(chain(['Page'], *splits[:i]))
        test_indices = list(chain(['Page'], splits[i + 1]))
        yield data[train_indices], data[test_indices]


def validate_forward_chaining(data, learner, folds):
    # type: (pd.DataFrame, Learner, int) -> float
    """Train and validate the model using forward chaining on the dataset."""
    scores = []
    for train, test in forward_chaining(data, folds):
        # Convert the test dataset into the regular format
        test = prepare_test_data(convert_to_test(test))

        prediction = learner.fit(train).predict(test)
        prediction.fillna(0, inplace=True)
        scores.append(smape(prediction['Actual'], prediction['Visits']))
    return np.mean(scores)


def validate_time_period(data, learner, start, end):
    # type: (pd.DataFrame, Learner, date, date) -> float
    date_columns = [date(*(int(x) for x in c.split('-')))
                    for c in get_date_columns(data)]

    assert start in date_columns, 'Starting date must be present in the data.'
    assert end in date_columns, 'Ending date must be present in the data.'

    training_dates = [c for c in date_columns if c < start]
    testing_dates = [c for c in date_columns if start <= c <= end]

    training_cols = ['%d-%02d-%02d' % (d.year, d.month, d.day) for d in training_dates]
    testing_cols = ['%d-%02d-%02d' % (d.year, d.month, d.day) for d in testing_dates]

    logger.info('Training on %d columns, testing on %d columns' %
                (len(training_cols), len(testing_cols)))

    train, test = data[['Page'] + training_cols], data[['Page'] + testing_cols]
    test = convert_to_test(test)
    test = prepare_test_data(test)

    prediction = learner.fit(train).predict(test)
    prediction.fillna(0, inplace=True)
    return smape(prediction['Actual'], prediction['Visits'])
