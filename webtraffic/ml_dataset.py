from os.path import join

import numpy as np
import pandas as pd

from data_provider import get_date_columns, TRAIN_DATA, DATA_DIR

ML_DATASET = join(DATA_DIR, 'ml_train_1.csv')


def prepare_ml_dataset(data, n_last_days, lag_days=30):
    date_columns = get_date_columns(data)
    used_data = data[['Page'] + date_columns[-n_last_days:]]

    flattened = pd.melt(used_data, id_vars='Page', var_name='date',
                        value_name='Visits')
    # For convenience, sort the data by pages
    flattened.sort_values(by='Page', inplace=True)
    flattened.reset_index(inplace=True, drop=True)
    # Drop any columns where the target value is unknown
    flattened.dropna(how='any', inplace=True)

    # Add lag columns to flattened
    lag_columns = list(
        reversed(['lag_%d' % i for i in range(1, lag_days + 1)]))
    flattened = flattened.reindex(
        columns=list(flattened.columns) + lag_columns)

    date_indices = {d: i for i, d in enumerate(date_columns)}

    # We will need the original data page indices and to set the index to page
    data['page_indices'] = data.index
    data.set_index('Page', inplace=True)

    page_indices = pd.DataFrame({
        'Page': flattened['Page'],
        'date_indices': flattened['date'].apply(date_indices.get),
    }).set_index('Page').join(data['page_indices']).reset_index()

    for lag in range(1, lag_days + 1):
        flattened['lag_%d' % lag] = data[date_columns].values[
            page_indices['page_indices'],
            page_indices['date_indices'] - lag
        ]

    # Since we're not lacking in training data, drop any row with NaN lag vars
    flattened.dropna(how='any', inplace=True)

    # Set correct dtypes
    flattened['date'] = flattened['date'].astype('datetime64[ns]')
    flattened['Visits'] = flattened['Visits'].astype(np.float64)
    flattened[lag_columns] = flattened[lag_columns].astype(np.float64)

    return flattened


if __name__ == '__main__':
    ds = pd.read_csv(TRAIN_DATA)
    ds = prepare_ml_dataset(ds, n_last_days=30)
    ds.to_csv(ML_DATASET)
