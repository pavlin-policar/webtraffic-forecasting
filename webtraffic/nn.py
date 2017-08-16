import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import MSELoss

from data_provider import TRAIN_DATA
from ml_dataset import ML_DATASET

N_EPOCHS = 1


def load_data():
    data = pd.read_csv(ML_DATASET)

    # # Add some extra helpful features
    def category(result):
        return pd.Series(result, dtype='category')

    data['day_of_week'] = category(data.date.dt.dayofweek)
    data['weekend'] = category(data.date.dt.dayofweek // 5 == 1)

    return data


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


data = load_data()
input()
sys.exit(0)
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

