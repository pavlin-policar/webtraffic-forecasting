import shutil
import sys
from os.path import join

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data.sampler import RandomSampler, SequentialSampler, \
    BatchSampler

from data_provider import MODELS_DIR, TEST_DATA, prepare_test_data, TRAIN_DATA, \
    get_date_columns
from ml_dataset import ML_VALIDATION, ML_TRAIN, get_info_file, LAG_DAYS, \
    lag_test_set_fname

N_EPOCHS = 20
BATCH_SIZE = 512


class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        hidden_size = 1000
        self.input = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        # self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.selu = nn.SELU()

    def forward(self, x):
        out = self.selu(self.input(x))
        out = self.selu(self.fc2(out))
        out = self.selu(self.fc3(out))
        # out = self.selu(self.fc4(out))
        out = self.output(out)
        return out


def compute_loss(data, y_data, criterion, model):
    losses = []
    for batch_idx in BatchSampler(SequentialSampler(data), BATCH_SIZE, False):
        x, y = data.iloc[batch_idx], y_data.iloc[batch_idx]
        x = Variable(torch.from_numpy(x.values), volatile=True).cuda()
        y = Variable(torch.from_numpy(y.values), volatile=True).cuda()

        y_hat = model(x)
        # Compute the loss and update the weights
        loss = criterion(y, y_hat)
        losses.append(loss.data[0])

    return np.mean(losses)


def save_checkpoint(state, is_best, filename='checkpoint.tar'):
    fname = join(MODELS_DIR, filename)
    torch.save(state, fname)
    if is_best:
        shutil.copyfile(fname, join(MODELS_DIR, 'model_best.tar'))


def load_data(data):
    data_info = get_info_file()

    # Normalize the data
    normalize_cols = data_info['normalize_columns']
    data[normalize_cols] -= data_info['mean']
    data[normalize_cols] /= data_info['std']

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

    # Remove columns that we won't use for training
    data.drop(['Page', 'title', 'date'], inplace=True, axis=1)

    # Prepare the data for training
    data = pd.get_dummies(data).astype(np.float32)

    y_data = data.pop('Visits')

    return data, y_data


def train_model():
    train, y_train = load_data(pd.read_csv(ML_TRAIN))
    val, y_val = load_data(pd.read_csv(ML_VALIDATION))

    model = NeuralNet(train.shape[1], 1).cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0)
    best_loss = np.inf

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

        training_loss = compute_loss(train, y_train, criterion, model)
        training_losses.append(training_loss)

        validation_loss = compute_loss(val, y_val, criterion, model)
        validation_losses.append(validation_loss)

        # Save checkpoint model and best model
        is_best = validation_loss < best_loss
        best_loss = min(validation_loss, best_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best)

        print('[%2d/%d] Training loss: %.4f - Validation loss: %.4f' %
              (epoch + 1, N_EPOCHS, training_loss, validation_loss))

    plt.plot(list(range(len(training_losses))), training_losses)
    plt.plot(list(range(len(validation_losses))), validation_losses)
    plt.show()
    sys.exit(0)


def make_lag_test_set(lag_days=LAG_DAYS):
    data = pd.read_csv(TRAIN_DATA)
    columns = ['Page'] + get_date_columns(data)[-lag_days:]
    data[columns].to_csv(lag_test_set_fname(lag_days), index=False)


def make_prediction():
    test_data = pd.read_csv(TEST_DATA)
    test_ids = dict(zip(test_data['Page'], test_data['Id']))
    # Extract date and page to own columns
    test_data['date'] = test_data['Page'].apply(lambda a: a[-10:])
    test_data['Page'] = test_data['Page'].apply(lambda a: a[:-11])

    test_dates = test_data['date'].unique()
    print(test_dates)
    test_data = pd.DataFrame(
        index=test_data['Page'].unique(),
        columns=test_dates,
    )

    lag_test_set = pd.read_csv(lag_test_set_fname(LAG_DAYS))
    data = lag_test_set.join(test_data, on='Page')

    # Free up some memory
    del test_data
    del lag_test_set

    # print(data)

    input()


if __name__ == '__main__':
    fire.Fire()
