import re
from os import listdir

import fire
import pandas as pd
from os.path import split, dirname, join, isdir

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


def combine_predictions(directory, name):
    # type: (str, str) -> None
    assert isdir(directory), 'The parameter must be a directory'

    predictions = pd.DataFrame(columns=['Id', 'Visits'])
    for file in listdir(directory):
        new_predictions = pd.read_csv(file)
        predictions = predictions.append(new_predictions, ignore_index=True)

    fname = join(PREDICTIONS_DIR, '%s.csv' % name)
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
    data['Date'] = data['Page'].apply(lambda a: a[-10:])
    # Remove the actual date from the `Page` field
    data['Page'] = data['Page'].apply(lambda a: a[:-11])

    return data


def get_language_dataset(dataset, language):
    # type: (str, str) -> str
    parts = split(dataset)
    new_fname = '%s_%s' % (language, parts[-1])
    return join(*parts[:-1], new_fname)


if __name__ == '__main__':
    fire.Fire()
