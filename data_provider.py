import re

import fire
import pandas as pd
from os.path import split

TRAIN_DATA = 'data/train_1.csv'
TEST_DATA = 'data/key_1.csv'

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
            '/'.join((*f_parts[:-1], new_fname)), index=False)


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


if __name__ == '__main__':
    fire.Fire()
