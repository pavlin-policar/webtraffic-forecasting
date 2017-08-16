import sys

import numpy as np
import pandas as pd
from torch import nn

from data_provider import TRAIN_DATA, get_language_dataset, \
    get_date_columns

N_EPOCHS = 1


def prepare_dataset(data, n_last_days=100, lag_days=30):
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
    lag_columns = list(reversed(['lag_%d' % i for i in range(1, lag_days + 1)]))
    flattened = flattened.reindex(columns=list(flattened.columns) + lag_columns)

    date_indices = {d: i for i, d in enumerate(date_columns)}

    # We will need the original data page indices and to set the index to page
    data['page_indices'] = data.index
    data.set_index('Page', inplace=True)

    page_indices = pd.DataFrame({
        'Page': flattened['Page'],
        'date_indices': flattened['date'].apply(date_indices.get),
    }).set_index('Page').join(data['page_indices']).reset_index()

    # Revert the data to its initial shape
    # data.drop('page_indices', inplace=True)
    # data.reset_index()

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

    # # Add some extra helpful features
    def category(result):
        return pd.Series(result, dtype='category')

    flattened['day_of_week'] = category(flattened.date.dt.dayofweek)
    flattened['weekend'] = category(flattened.date.dt.dayofweek // 5 == 1)

    return flattened


data = pd.read_csv(get_language_dataset(TRAIN_DATA, 'de'))
data = prepare_dataset(data, n_last_days=10)
print(data)
# data.to_csv('ml_data_last_10_days_30_lag.csv')

sys.exit(0)


class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out


model = LinearRegression(39, 1).cuda()

criterion = MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.1)


data = pd.read_csv(get_language_dataset(TRAIN_DATA, 'en'))
# Normalize the data
cols = data.columns.difference(['Page'])
data[cols] = (data[cols] - data[cols].mean()) / data[cols].std()


losses = []
columns = None
for i in range(N_EPOCHS):
    try:
        for idx, minibatch in enumerate(epoch(data, batch_size=256)):
            targets = minibatch.pop('Visits')
            minibatch = minibatch.drop(['Page', 'date'], axis=1)
            columns = minibatch.columns

            data = pd.get_dummies(minibatch)
            data = data.values.astype(np.float32)
            targets = targets.values.astype(np.float32)

            data = Variable(torch.from_numpy(data)).cuda()
            targets = Variable(torch.from_numpy(targets)).cuda()

            # Reset gradients
            optimizer.zero_grad()

            # Compute the predictions
            outputs = model(data)

            # Compute the loss and updatae the weights
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            losses.append((idx, np.log(loss.data[0])))

            print('Epoch [%d/%d], Loss: %.8f' % (i + 1, N_EPOCHS, loss.data[0]))

    except KeyboardInterrupt:
        print()
        # for v in sorted(
        #         list(zip(columns, list(model.parameters())[0][0].data)),
        #         key=lambda x: abs(x[1]), reverse=True):
        #     print("%20s: %.4f" % v)
        plt.plot(*list(zip(*losses)))
        plt.show()
        sys.exit(0)

