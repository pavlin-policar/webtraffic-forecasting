from os.path import join

import pandas as pd

from data_provider import TRAIN_DATA, PREDICTIONS_DIR, TEST_DATA

DAYS_TO_CONSIDER = 49

train = pd.read_csv(TRAIN_DATA)
test = pd.read_csv(TEST_DATA)

# Since the `Page` field looks something like this:
# !vote_en.wikipedia.org_all-access_all-agents_2017-01-01
# we need to be able to match the page, we remove the date from that. We don't
# want to lose the date, so we extract that to a separate column.
# test['Date'] = test.Page.apply(lambda a: a[-10:])
test['Page'] = test.Page.apply(lambda a: a[:-11])

# Take the last 14 days and compute the median of that
train['Visits'] = train[train.columns[-DAYS_TO_CONSIDER:]].median(axis=1, skipna=True)

test = test.merge(train[['Page', 'Visits']], how='left')
test.loc[test['Visits'].isnull(), 'Visits'] = 0

test[['Id', 'Visits']].to_csv(
    join(PREDICTIONS_DIR, 'median_%d.csv') % DAYS_TO_CONSIDER, index=False)
