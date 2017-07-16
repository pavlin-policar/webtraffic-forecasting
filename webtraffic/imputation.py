import numpy as np
import pandas as pd

from data_provider import TRAIN_DATA, get_language_dataset


def sliding_window_median_imputation(data, window_size=1):
    data = data.astype(np.float64)

    while np.isnan(data).any():
        new_data = data.copy()
        for index, nan in enumerate(np.isnan(data)):
            if nan:
                neigbours = data[max(0, index - window_size):min(
                    index + window_size + 1, len(data))]
                new_data[index] = np.nanmedian(neigbours)
        data = new_data

    return data.astype(np.object)


def perform_imputation(data, imputation):
    data.iloc[:, 1:] = data.iloc[:, 1:].apply(imputation)
    return data


if __name__ == '__main__':
    train = pd.read_csv(get_language_dataset(TRAIN_DATA, 'en'))
    train.iloc[:, 1:] = train.iloc[:, 1:].apply(sliding_window_median_imputation)
    print(train.iloc)
