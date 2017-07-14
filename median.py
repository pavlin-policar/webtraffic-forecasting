import pandas as pd


train = pd.read_csv('data/train_1.csv')
test = pd.read_csv('data/key_1.csv')

test['Page'] = test.Page.apply(lambda a: a[:-11])

train['Visits'] = train[train.columns[-14:]].median(axis=1, skipna=True)

test = test.merge(train[['Page', 'Visits']], how='left')
test.loc[test.Visits.isnull(), 'Visits'] = 0

test[['Id', 'Visits']].to_csv('predictions/median.csv', index=False)
