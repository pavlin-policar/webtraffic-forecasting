import random
import re
from datetime import datetime
from os import listdir
from os.path import split, dirname, join, isdir
from typing import Union, List

import fire
import numpy as np
import pandas as pd

DATA_DIR = join(dirname(dirname(__file__)), 'data')
MODELS_DIR = join(dirname(dirname(__file__)), 'models')
PREDICTIONS_DIR = join(dirname(dirname(__file__)), 'predictions')

TRAIN_DATA = join(DATA_DIR, 'train_1.csv')
TEST_DATA = join(DATA_DIR, 'key_1.csv')

__LANG_REGEX = re.compile(r'([a-z]{2})\.wikipedia\.org')


def get_page_language(page):
    res = __LANG_REGEX.search(page)
    return res.group(1) if res else 'na'


def split_by_language(fname):
    # type: (str) -> None
    """Split a given file into multiple csv files by language."""
    data = pd.read_csv(fname)  # type: pd.DataFrame
    data['Language'] = data['Page'].map(get_page_language)

    for lang, group in data.groupby('Language'):
        f_parts = split(fname)
        new_fname = '%s_%s' % (lang, f_parts[-1])
        group.drop(['Language'], axis=1).to_csv(
            join(*f_parts[:-1], new_fname), index=False)


def save_predictions(predictions, fname):
    # type: (pd.DataFrame, str) -> None
    assert 'Id' in predictions.columns and 'Visits' in predictions.columns, \
        'Data must contain the `Id` and `Visits` columns.'
    predictions[['Id', 'Visits']].to_csv(fname, index=False)


def combine_predictions(directory, name='combined_predictions'):
    # type: (str, str) -> None
    assert isdir(directory), 'The parameter must be a directory'

    predictions = pd.DataFrame(columns=['Id', 'Visits'])
    for file in listdir(directory):
        new_predictions = pd.read_csv(join(directory, file))
        predictions = predictions.append(new_predictions, ignore_index=True)

    fname = join(directory, '%s.csv' % name)
    predictions.to_csv(fname, index=False)


def convert_to_test(data):
    # type: (pd.DataFrame) -> pd.DataFrame
    """Convert a dataframe to the format used in the test data."""
    test = pd.DataFrame(columns=['Page', 'Actual'])

    for column in data.drop('Page', axis=1).columns:
        new_df = data[['Page', column]]
        new_df['Page'] = new_df['Page'].apply(lambda a: '%s_%s' % (a, column))
        new_df.columns = ['Page', 'Actual']
        test = test.append(new_df, ignore_index=True)

    return test


def prepare_test_data(data):
    # type: (pd.DataFrame) -> pd.DataFrame
    # Extract the date from the `Page` field and store it into `Date`
    data['date'] = data['Page'].apply(lambda a: a[-10:])
    data['date'] = data['date'].astype('datetime64[ns]')
    # Remove the actual date from the `Page` field
    data['Page'] = data['Page'].apply(lambda a: a[:-11])

    return data


def get_language_dataset(dataset, language):
    # type: (str, str) -> str
    parts = split(dataset)
    new_fname = '%s_%s' % (language, parts[-1])
    return join(*parts[:-1], 'langs', new_fname)


def get_date_columns(data):
    # type: (pd.DataFrame) -> List[str]
    """Get all the date columns in the data in ascending order."""
    dates = sorted(datetime.strptime(c, '%Y-%m-%d') for c in data.columns
                   if re.match(r'\d{4}-\d{2}-\d{2}', c))
    return date_to_str(dates)


def str_to_date(dates):
    # type: (Union[str, List[str]]) -> List[datetime]
    if not isinstance(dates, list):
        dates = [dates]

    return [datetime.strptime(date, '%Y-%m-%d') for date in dates]


def date_to_str(dates):
    # type: (Union[datetime, List[datetime]]) -> List[str]
    if not isinstance(dates, list):
        dates = [dates]

    return [date.strftime('%Y-%m-%d') for date in dates]


def epoch(data, batch_size=128, lag_length=30):
    """Create a minibatch from the data."""
    all_dates = get_date_columns(data)
    # We will ignore the lag days since we don't want to have null variables
    dates = all_dates[lag_length:]

    # Prepare the columns that we will set during the initial iteration
    columns = ['Page', 'date', 'Visits']
    lag_columns = list(reversed(['lag_%d' % i for i in range(1, lag_length + 1)]))
    columns += lag_columns

    candidates = {d: list(dates) for d in data.index}

    while len(candidates):
        sample_indices = random.choices(list(candidates.keys()), k=batch_size)
        samples = []
        print(len(candidates))

        for idx in sample_indices:
            instance = data.loc[idx]
            date = random.choice(candidates[idx])
            candidates[idx].remove(date)

            # If the sample has no more valid dates, remove the sample from the
            # sample pool
            if not len(candidates[idx]):
                del candidates[idx]

            # Begin preparing the minibatch dataframes
            if pd.isnull(instance[date]):
                continue

            sample = pd.Series(index=columns)
            sample['Page'] = instance['Page']
            sample['date'] = date
            sample['Visits'] = instance[date]

            # Impute any missing values in the row once we know that the date
            # we chose indeed has a known value
            instance = instance.fillna(instance.rolling(window=5).median())
            # Fill the row with lag variables
            date_index = all_dates.index(date)
            date_range = all_dates[date_index - lag_length:date_index]
            sample.loc[lag_columns] = instance[date_range].values

            samples.append(sample)

        # Merge all the sample dataframes into a single minibatch dataframe
        minibatch = pd.DataFrame(samples)

        # Set the correct dtypes
        minibatch['date'] = pd.to_datetime(minibatch['date'], format='%Y-%m-%d')
        minibatch['Visits'] = minibatch['Visits'].astype(np.float64)
        minibatch[lag_columns] = minibatch[lag_columns].astype(np.float64)

        # Add some extra helpful features
        def category(result):
            return pd.Series(result, dtype='category')

        minibatch['day_of_week'] = category(minibatch.date.dt.dayofweek)
        minibatch['weekend'] = category(minibatch.date.dt.dayofweek // 5 == 1)

        yield minibatch


if __name__ == '__main__':
    fire.Fire()
