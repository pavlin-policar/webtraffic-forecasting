import random

import numpy as np
import pandas as pd

from data_provider import TRAIN_DATA, get_date_columns


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


all_data = pd.read_csv(TRAIN_DATA)


for mb in epoch(all_data, batch_size=256):
    print(mb)
    break
