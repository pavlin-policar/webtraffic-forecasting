import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import MSELoss
from torch.utils.data.sampler import RandomSampler

from ml_dataset import ML_DATASET

N_EPOCHS = 10
BATCH_SIZE = 256


def load_data():
    data = pd.read_csv(ML_DATASET)

    # Set correct dtypes
    data['date'] = data['date'].astype('datetime64[ns]')

    # # Add some extra helpful features
    def category(result):
        return pd.Series(result, dtype='category')

    data['day_of_week'] = category(data.date.dt.dayofweek)
    data['weekend'] = category(data.date.dt.dayofweek // 5 == 1)

    return data


class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


class NeuralNet(nn.Module):
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


data = load_data()

# Normalize the data
normalize_cols = ['Visits'] + [c for c in data.columns if 'lag' in c]
mean = data[normalize_cols].mean().mean()
std = data[normalize_cols].values.std(ddof=1)
data[normalize_cols] -= mean
data[normalize_cols] /= std

# Remove columns
data.drop(['Page', 'date'], inplace=True, axis=1)

# Prepare the dataset for training
target = data.pop('Visits').astype(np.float32)

data = pd.get_dummies(data).astype(np.float32)


class BatchSampler:
    """Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(range(10), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(range(10), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


model = LinearRegression(data.shape[1], 1).cuda()
criterion = MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.1)



losses = []
columns = None


def done():
    print()
    columns = data.columns
    for v in sorted(
            list(zip(columns, list(model.parameters())[0][0].data)),
            key=lambda x: abs(x[1]), reverse=True):
        print("%20s: %.4f" % v)
    plt.plot(list(range(len(losses))), losses)
    plt.show()
    sys.exit(0)


try:
    for epoch in range(N_EPOCHS):
        for batch_idx in BatchSampler(RandomSampler(data), BATCH_SIZE, False):
            x, y = data.iloc[batch_idx], target.iloc[batch_idx]

            x = Variable(torch.from_numpy(x.values)).cuda()
            y = Variable(torch.from_numpy(y.values)).cuda()

            # Reset gradients
            optimizer.zero_grad()

            # Compute the predictions
            y_hat = model(x)

            # Compute the loss and updatae the weights
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

        print(loss)
        losses.append(loss.data[0])

except KeyboardInterrupt:
    done()
finally:
    done()
