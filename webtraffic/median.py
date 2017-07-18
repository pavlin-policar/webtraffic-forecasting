from os.path import join

import numpy as np
import pandas as pd
from functools import partial

from data_provider import TRAIN_DATA, PREDICTIONS_DIR, TEST_DATA
from imputation import sliding_window_median_imputation, perform_imputation

DAYS_TO_CONSIDER = 49

train = pd.read_csv(TRAIN_DATA)

train_flattened = pd.melt(
    train[list(train.columns[-49:]) + ['Page']],
    id_vars='Page', var_name='date', value_name='Visits')
train_flattened['date'] = train_flattened['date'].astype('datetime64[ns]')
train_flattened['weekend'] = (train_flattened.date.dt.dayofweek // 5 == 1).astype(float)

test = pd.read_csv(TEST_DATA)

# Since the `Page` field looks something like this:
# !vote_en.wikipedia.org_all-access_all-agents_2017-01-01
# we need to be able to match the page, we remove the date from that. We don't
# want to lose the date, so we extract that to a separate column.
test['date'] = test.Page.apply(lambda a: a[-10:])
test['date'] = test['date'].astype('datetime64[ns]')
test['weekend'] = (test.date.dt.dayofweek // 5 == 1).astype(float)
test['Page'] = test.Page.apply(lambda a: a[:-11])

train_page_per_dow = train_flattened.groupby(['Page', 'weekend']).median().reset_index()


# imputation = partial(sliding_window_median_imputation, window_size=2)
# train = perform_imputation(train, imputation)

# Take the last 14 days and compute the median of that
train['Visits'] = train[train.columns[-DAYS_TO_CONSIDER:]].median(axis=1, skipna=True)

test = test.merge(train_page_per_dow[['Page', 'Visits']], how='left')
test.loc[test['Visits'].isnull(), 'Visits'] = 0

test[['Id', 'Visits']].to_csv(
    join(PREDICTIONS_DIR, 'median_weekend_%d.csv') % DAYS_TO_CONSIDER,
    index=False,
)
