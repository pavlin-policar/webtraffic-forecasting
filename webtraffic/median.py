import pandas as pd

DAYS_TO_CONSIDER = 49

train = pd.read_csv('data/train_1.csv')
train.fillna(0, inplace=True)

test = pd.read_csv('data/key_1.csv')

# Since the `Page` field looks something like this:
# !vote_en.wikipedia.org_all-access_all-agents_2017-01-01
# we need to be able to match the page, we remove the date from that. We don't
# want to lose the date, so we extract that to a separate column.
test['Date'] = test.Page.apply(lambda a: a[-10:])
test['Page'] = test.Page.apply(lambda a: a[:-11])

# Take the last 14 days and compute the median of that
train['Visits'] = train[train.columns[-DAYS_TO_CONSIDER:]].median(axis=1)

test = test.merge(train[['Page', 'Visits']], how='left')

test[['Id', 'Visits']].to_csv(
    'predictions/median_%d.csv' % DAYS_TO_CONSIDER, index=False)
