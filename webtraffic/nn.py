import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data.sampler import RandomSampler, SequentialSampler, \
    BatchSampler

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

    data[['title', 'lang', 'access', 'agent']] = data['Page'].str.extract(
        '(.+)_(\w{2})\.wikipedia\.org_([^_]+)_([^_]+)')
    data['lang'] = data['lang'].astype('category')
    data['access'] = data['access'].astype('category')
    data['agent'] = data['agent'].astype('category')

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
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, output_size)
        self.selu = nn.SELU()

    def forward(self, x):
        out = self.selu(self.fc1(x))
        out = self.selu(self.fc2(out))
        out = self.fc3(out)
        return out


data = load_data()

# Normalize the data
normalize_cols = ['Visits'] + [c for c in data.columns if 'lag' in c]
mean = data[normalize_cols].mean().mean()
std = data[normalize_cols].values.std(ddof=1)
data[normalize_cols] -= mean
data[normalize_cols] /= std

# Remove columns that we won't use for training
data.drop(['Page', 'title', 'date'], inplace=True, axis=1)

# Prepare the data for training
data = pd.get_dummies(data).astype(np.float32)

# Create the train/validation/test splits
train, validation, test = np.split(
    data.sample(frac=1), [int(.6 * len(data)), int(.8 * len(data))])

y_train = train.pop('Visits').astype(np.float32)
y_validation = validation.pop('Visits').astype(np.float32)
y_test = test.pop('Visits').astype(np.float32)


def compute_loss(data, criterion, model):
    losses = []
    for batch_idx in BatchSampler(SequentialSampler(data), BATCH_SIZE, False):
        x, y = data.iloc[batch_idx], data.iloc[batch_idx]
        x = Variable(torch.from_numpy(x.values)).cuda()
        y = Variable(torch.from_numpy(y.values)).cuda()

        y_hat = model(x)
        # Compute the loss and update the weights
        loss = criterion(y, y_hat)
        print(loss)
        loss.backward()
        losses.append(loss.data[0])

    print(losses)
    return np.mean(losses)


model = LinearRegression(train.shape[1], 1).cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.1)

training_losses, validation_losses = [], []

for epoch in range(N_EPOCHS):
    # Train on minibatches
    model.train(True)
    for batch_idx in BatchSampler(RandomSampler(train), BATCH_SIZE, False):
        x, y = train.iloc[batch_idx], y_train.iloc[batch_idx]
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

    # Compute the training and validation loss
    model.eval()

    print('training loss')
    training_loss = compute_loss(train, criterion, model)
    training_losses.append(training_loss)

    print('validation loss')
    validation_loss = compute_loss(validation, criterion, model)
    validation_losses.append(validation_loss)

    print('[%2d/%d] Training loss: %.4f Validation loss: %.4f' %
          (epoch, N_EPOCHS, training_loss, validation_loss))

print()
plt.plot(list(range(len(training_losses))), training_losses)
plt.plot(list(range(len(validation_losses))), validation_losses)
plt.show()
sys.exit(0)
