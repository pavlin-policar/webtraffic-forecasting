import random

import numpy as np
import pandas as pd

from data_provider import TRAIN_DATA, get_date_columns


def fetch_minibatch(data, batch_size=128, lag_length=30):
    """Create a minibatch from the data."""
    all_dates = get_date_columns(data)
    # We will ignore the lag days since we don't want to have null variables
    dates = all_dates[lag_length:]

    columns = ['Page', 'date', 'Visits']
    lag_columns = list(reversed(['lag_%d' % i for i in range(1, lag_length + 1)]))
    columns += lag_columns
    minibatch = pd.DataFrame(columns=columns)

    for idx, (_, instance) in enumerate(data.sample(batch_size).iterrows()):

        date, dates_ = random.choice(dates), list(dates)
        # We don't want to deal with NaN values, so skip over those
        # Try to select a non-NaN target and if no such value exists, skip row
        while pd.isnull(instance[date]) and len(dates_):
            dates_.remove(date)
            date = random.choice(dates_)
        if pd.isnull(instance[date]):
            continue

        minibatch.loc[idx, 'Page'] = instance['Page']
        minibatch.loc[idx, 'date'] = date
        minibatch.loc[idx, 'Visits'] = instance[date]

        # Impute any missing values in the row once we know that the date we
        # chose indeed has a known value
        instance = instance.fillna(instance.rolling(window=5).median())
        # Fill the row with lag variables
        date_index = all_dates.index(date)
        date_range = all_dates[date_index - lag_length:date_index]

        minibatch.loc[idx, lag_columns] = instance[date_range].values

    # Set the correct dtypes
    minibatch['date'] = minibatch['date'].astype('datetime64[ns]')
    minibatch['Visits'] = minibatch['Visits'].astype(np.float64)
    minibatch[lag_columns] = minibatch[lag_columns].astype(np.float64)

    # Add some extra helpful features
    def category(result):
        return pd.Series(result, dtype='category')

    minibatch['day_of_week'] = category(minibatch.date.dt.dayofweek)
    minibatch['weekend'] = category(minibatch.date.dt.dayofweek // 5 == 1)

    return minibatch


all_data = pd.read_csv(TRAIN_DATA)
mb = fetch_minibatch(all_data, batch_size=1)
print(mb)
