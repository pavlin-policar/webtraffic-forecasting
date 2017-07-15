from itertools import chain

import numpy as np
import pandas as pd

from models import LastNDaysMedianLearner, Learner
from data_provider import convert_to_test, prepare_test_data, TRAIN_DATA


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


def validate_last_days(data, learner, n_days=60):
    # type: (pd.DataFrame, Learner, Optional[int]) -> float
    """Train and validate the model on the last `n` days of the dataset."""
    # Training data from first column forth, to skip the `Page` column
    train, test = data[data.columns[1:-n_days]], data[data.columns[-n_days:]]
    # Add the `Page` column to both train and test data
    train['Page'] = test['Page'] = data['Page']

    test = convert_to_test(test)
    test = prepare_test_data(test)

    prediction = learner.fit(train).predict(test)
    return smape(prediction['Actual'], prediction['Predicted'])


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
        scores.append(smape(prediction['Actual'], prediction['Predicted']))
    return np.mean(scores)


if __name__ == '__main__':
    train = pd.read_csv(TRAIN_DATA)
    train.fillna(0, inplace=True)
    for days in range(14, 100, 7):
        print('SMAPE (%2d days): %.2f' % (days, validate_last_days(
            train, LastNDaysMedianLearner(days_to_consider=days))))
