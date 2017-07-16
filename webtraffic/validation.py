from collections import defaultdict
from itertools import chain

import numpy as np
import pandas as pd

from models import LastNDaysMedianLearner, Learner
from data_provider import convert_to_test, prepare_test_data, TRAIN_DATA, \
    get_language_dataset


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
    prediction.fillna(0, inplace=True)
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
    langs = ['de', 'en', 'es', 'fr', 'ja', 'na', 'ru', 'zh']
    for days in range(56, 100, 7):
        scores = []
        for lang in langs:
            train = pd.read_csv(get_language_dataset(TRAIN_DATA, lang))
            score = validate_forward_chaining(
                train, LastNDaysMedianLearner(days_to_consider=days), 5)
            print('%s SMAPE (%2d days): %.2f' % (lang, days, score))
            scores.append(score)
        print('SMAPE (%2d days): %.2f\n' % (days, np.mean(scores)))